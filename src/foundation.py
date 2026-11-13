"""
foundation.py
=============
Zero-shot wrappers for pretrained time-series forecasters, plus Prophet.

Every wrapper takes a list of context windows (one per forecast origin, each
ending at the origin) and returns the point forecast at horizon h, so a whole
walk-forward can be scored in one batched call. Past-only covariates are
optional: pass one (k x T) array per context, aligned with that context.

  TimesFM 2.5  google/timesfm-2.5-200m-pytorch   univariate
  TimesFM 3    google/timesfm-3.0-pytorch        univariate or + past covariates
  Chronos-2    amazon/chronos-2                  univariate or + past covariates
  Prophet      refit at every origin, no seasonality (inputs are already SA)

Weights download from Hugging Face on first use (~1.5 GB total).
"""
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class TimesFM25:
    name = "TimesFM-2.5"

    def __init__(self, max_context=1024, max_horizon=64):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.m.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))

    def predict(self, contexts, h, covariates=None, batch=64):
        out = []
        for i in range(0, len(contexts), batch):
            pt, _ = self.m.forecast(horizon=h, inputs=[np.asarray(c, float) for c in contexts[i:i+batch]])
            out.append(pt[:, h-1])
        return np.concatenate(out)


class TimesFM3:
    name = "TimesFM-3"

    def __init__(self):
        import timesfm
        self.m = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")

    def predict(self, contexts, h, covariates=None, batch=256):
        out = []
        for i in range(0, len(contexts), batch):
            cov = covariates[i:i+batch] if covariates is not None else None
            res = self.m.predict_batch(contexts=[np.asarray(c, np.float32) for c in contexts[i:i+batch]],
                                       horizon=h, past_only_covariates=cov)
            out += [float(np.atleast_2d(r.forecast)[0, h-1]) for r in res]
        return np.array(out)


class Chronos2:
    name = "Chronos-2"

    def __init__(self):
        import torch
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu",
                                                     torch_dtype=torch.float32)

    def predict(self, contexts, h, covariates=None, batch=64):
        out = []
        for i in range(0, len(contexts), batch):
            inp = []
            for j, c in enumerate(contexts[i:i+batch]):
                d = {"target": np.asarray(c, np.float32)}
                if covariates is not None:
                    cv = np.atleast_2d(covariates[i+j]).astype(np.float32)
                    d["past_covariates"] = {f"x{k}": cv[k] for k in range(cv.shape[0])}
                inp.append(d)
            _, mean = self.p.predict_quantiles(inp, prediction_length=h, quantile_levels=[0.5])
            out += [float(m[0, h-1]) for m in mean]
        return np.array(out)


class ProphetModel:
    name = "Prophet"

    def __init__(self, freq):
        from prophet import Prophet  # noqa: F401  (import first: it installs its own log handlers)
        for name in ("cmdstanpy", "prophet"):
            lg = logging.getLogger(name); lg.setLevel(logging.ERROR); lg.handlers.clear()
            lg.addHandler(logging.NullHandler()); lg.propagate = False
        self.freq = freq

    def forecast_path(self, series: pd.Series, h):
        """Forecasts for steps 1..h after the last observation."""
        from prophet import Prophet
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    uncertainty_samples=0)
        m.fit(pd.DataFrame({"ds": series.index, "y": series.values}))
        fut = m.make_future_dataframe(periods=h, freq=self.freq, include_history=False)
        return m.predict(fut)["yhat"].to_numpy()

    def predict_one(self, series: pd.Series, h):
        return float(self.forecast_path(series, h)[h-1])
