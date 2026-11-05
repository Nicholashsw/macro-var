"""
phase3_foundation.py
====================
Do pretrained time-series foundation models beat AR(BIC) on the Phase-1 core?

Same sample, targets, horizons and expanding-window origins as Phase 2
(1994Q2 onward, OOS from 55%, h = 1 and 4, targets GDP / INF / POLICY / FX).
Every model forecasts the same origins, so every loss vector lines up.

Models
  benchmarks   RW, AR(BIC)
  references   OLS-VAR(2) and BVAR-Tight on {GDP, INF, POLICY, FX} + OIL exog
  zero-shot    TimesFM 2.5, TimesFM 3, Chronos-2   (each target alone)
               TimesFM 3 + cov, Chronos-2 + cov    (other core variables + OIL
                                                    as past-only covariates)
  Prophet      univariate, refit at every origin
  averages     equal-weight combinations, formed after the fact from the
               forecasts above (no estimated weights, so nothing to leak)

Scoring: relRMSE vs RW (Phase-2 convention) and vs AR(BIC); Diebold-Mariano vs
AR(BIC) with HLN correction; joint 90% MCS per target at h = 1; pre-2020 vs
2020+ split.

Caveat: the foundation models were pretrained on large public corpora that
almost certainly contain these FRED series, and every origin here predates
their release. Read their scores as an upper bound.
Research/education.
"""
import os, json, warnings, subprocess, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

from phase2_modelclass import build_design, bvar_B, iterate_B
from foundation import TimesFM25, TimesFM3, Chronos2, ProphetModel

ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "fred_data"; DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)

P = 2; MAXLAG_AR = 8; H_LIST = [1, 4]; OOS_START_FRAC = 0.55
TARGETS = ["GDP", "INF", "POLICY", "FX"]; EXOG = ["OIL"]
SPEC = {"GDP":("GDPC1","growth"), "INF":("PCEPILFE","growth"), "POLICY":("FEDFUNDS","level"),
        "FX":("RBUSBIS","growth"), "INV":("GPDIC1","growth"), "CONS":("PCECC96","growth"),
        "FISCAL":("GCEC1","growth"), "EXPORTS":("EXPGSC1","growth"), "WEALTH":("TNWBSHNO","growth"),
        "HOUSING":("USSTHPI","growth"), "OIL":("WTISPLC","oilret"), "TFP":("FERNALD_TFPUTIL","level")}


# ---------------------------------- data -------------------------------------
def ensure(sid):
    f = DATA / f"{sid}.csv"
    if f.exists(): return
    if sid == "FERNALD_TFPUTIL":
        xl = DATA / "fernald_tfp.xlsx"
        subprocess.run(["curl", "-sSL", "--max-time", "90", "-o", str(xl),
                        "https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx"], check=True)
        df = pd.read_excel(xl, sheet_name="quarterly", header=1)[["date", "dtfp_util"]]
        df = df[df["date"].astype(str).str.match(r"^\d{4}:Q[1-4]$")].copy()
        q = {"1": 1, "2": 4, "3": 7, "4": 10}
        df["date"] = df["date"].map(lambda s: pd.Timestamp(int(str(s)[:4]), q[str(s)[-1]], 1))
        df["dtfp_util"] = pd.to_numeric(df["dtfp_util"], errors="coerce")
        df.dropna().set_index("date").sort_index().to_csv(f, header=["value"])
        return
    subprocess.run(["curl", "-sS", "--max-time", "60", "-o", str(f),
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"], check=True)

def load_q(sid):
    df = pd.read_csv(DATA / f"{sid}.csv"); df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"]); df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index().resample("QS").mean()

def tf(lv, k):
    return lv if k == "level" else (400.0*np.log(lv).diff() if k == "growth" else 100.0*np.log(lv).diff())


def ar_bic_path(y, h, maxlag=MAXLAG_AR):
    y = np.asarray(y, float); n = len(y); best = (np.inf, 1, None)
    for p in range(1, maxlag+1):
        if n-p < p+5: break
        Y = y[p:]; X = np.column_stack([np.ones(n-p)] + [y[p-1-j:n-1-j] for j in range(p)])
        b, *_ = np.linalg.lstsq(X, Y, rcond=None); r = Y - X@b; s2 = max(r@r/len(Y), 1e-12)
        bic = len(Y)*np.log(s2) + (p+1)*np.log(len(Y))
        if bic < best[0]: best = (bic, p, b)
    _, p, b = best; hist = list(y[-p:]); out = []
    for _ in range(h):
        x = np.array([1.0] + [hist[-1-j] for j in range(p)]); v = float(x@b); out.append(v); hist.append(v)
    return np.array(out)


def dm_test(e1, e2, h):
    """DM on squared loss, Newey-West (h-1 lags), HLN small-sample correction.
    Negative stat: model 1 has lower loss."""
    d = e1**2 - e2**2; n = len(d); dc = d - d.mean()
    v = dc@dc/n + sum(2*(1-l/h)*(dc[l:]@dc[:-l])/n for l in range(1, h))
    if v <= 0: return np.nan, np.nan
    s = d.mean()/np.sqrt(v/n) * np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return float(s), float(2*stats.t.sf(abs(s), n-1))


# ================================= main ======================================
if __name__ == "__main__":
    t0 = time.time()
    for _, (sid, _) in SPEC.items(): ensure(sid)
    panel = pd.DataFrame({n: tf(load_q(sid), k) for n, (sid, k) in SPEC.items()}).dropna().loc["1994-04-01":]
    N = len(panel); i0 = int(N*OOS_START_FRAC); h_max = max(H_LIST)
    origins = list(range(i0, N-h_max))
    dates = panel.index
    print(f"[panel] {N} q  {dates.min().date()}->{dates.max().date()}  "
          f"oos from {dates[i0].date()}  {len(origins)} origins\n")

    endog = TARGETS; delta = np.array([1.0 if SPEC[x][1] == "level" else 0.0 for x in endog])
    Y = panel[endog].values; Xo = panel[EXOG].values
    actual = {(t, h): np.array([panel[t].values[i+h] for i in origins]) for t in TARGETS for h in H_LIST}
    fc = {}   # (model, target, h) -> array over origins

    # benchmarks
    for t in TARGETS:
        y = panel[t].values
        for h in H_LIST:
            fc[("RW", t, h)] = np.array([y[i] for i in origins])
            fc[("AR(BIC)", t, h)] = np.array([ar_bic_path(y[:i+1], h)[-1] for i in origins])

    # VAR references, iterated with AR(BIC)-projected exog, as in Phase 2
    for label, lam in [("OLS-VAR", None), ("BVAR-Tight", 0.1)]:
        rows = {h: [] for h in H_LIST}
        for i in origins:
            en, ex = Y[:i+1], Xo[:i+1]
            exf = {h: np.column_stack([ar_bic_path(panel[e].values[:i+1], h) for e in EXOG]) for h in H_LIST}
            if lam is None:
                Yd, Xd, n, m = build_design(en, ex, P); B, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
            else:
                B, n, m = bvar_B(en, ex, P, lam, delta)
            for h in H_LIST: rows[h].append(iterate_B(B, en, exf[h], P, n, m, h)[-1])
        for h in H_LIST:
            arr = np.array(rows[h])
            for j, t in enumerate(TARGETS): fc[(label, t, h)] = arr[:, j]
    print(f"  benchmarks + VARs   {time.time()-t0:5.0f}s", flush=True)

    # Prophet
    pr = ProphetModel("QS")
    for t in TARGETS:
        s = panel[t]
        for h in H_LIST:
            fc[("Prophet", t, h)] = np.array([pr.predict_one(s.iloc[:i+1], h) for i in origins])
    print(f"  prophet             {time.time()-t0:5.0f}s", flush=True)

    # foundation models: one batched call per (model, target, h)
    ctx = {t: [panel[t].values[:i+1] for i in origins] for t in TARGETS}
    cov = {t: [panel[[c for c in TARGETS if c != t] + EXOG].values[:i+1].T for i in origins] for t in TARGETS}
    for model, with_cov in [(TimesFM25(max_context=512, max_horizon=16), [False]),
                            (TimesFM3(), [False, True]),
                            (Chronos2(), [False, True])]:
        for use in with_cov:
            label = model.name + (" + cov" if use else "")
            for t in TARGETS:
                for h in H_LIST:
                    fc[(label, t, h)] = model.predict(ctx[t], h, covariates=cov[t] if use else None)
            print(f"  {label:18s}  {time.time()-t0:5.0f}s", flush=True)

    FMS = ["TimesFM-2.5", "TimesFM-3", "TimesFM-3 + cov", "Chronos-2", "Chronos-2 + cov"]
    COMBOS = {"Avg: BVAR + TimesFM-3 cov": ["BVAR-Tight", "TimesFM-3 + cov"],
              "Avg: foundation models": FMS,
              "Avg: AR, BVAR, FM + cov": ["AR(BIC)", "BVAR-Tight", "TimesFM-3 + cov", "Chronos-2 + cov"]}
    for name, members in COMBOS.items():
        for t in TARGETS:
            for h in H_LIST:
                fc[(name, t, h)] = np.mean([fc[(m, t, h)] for m in members], axis=0)

    models = list(dict.fromkeys(k[0] for k in fc))
    family = lambda m: ("benchmark" if m in ("RW", "AR(BIC)") else "average" if m.startswith("Avg")
                        else "var" if "VAR" in m else "prophet" if m == "Prophet" else "foundation")
    pre = np.array([dates[i] < pd.Timestamp("2020-01-01") for i in origins])

    def rel(e, eb, mask=None):
        mask = np.ones(len(e), bool) if mask is None else mask
        return float(np.sqrt((e[mask]**2).mean()/(eb[mask]**2).mean()))

    rows = []
    for m in models:
        for t in TARGETS:
            for h in H_LIST:
                a = actual[(t, h)]; e = a - fc[(m, t, h)]
                e_rw = a - fc[("RW", t, h)]; e_ar = a - fc[("AR(BIC)", t, h)]
                dm, p = dm_test(e, e_ar, h) if m != "AR(BIC)" else (np.nan, np.nan)
                rows.append({"model": m, "family": family(m), "target": t, "h": h, "n": len(e),
                             "rmse": float(np.sqrt((e**2).mean())),
                             "relRMSE_RW": rel(e, e_rw), "relRMSE_AR": rel(e, e_ar),
                             "dm_vs_AR": dm, "dm_p": p,
                             "relRMSE_AR_pre2020": rel(e, e_ar, pre),
                             "relRMSE_AR_2020on": rel(e, e_ar, ~pre)})
    long = pd.DataFrame(rows).round(4)

    wide = long.pivot_table(index=["model", "family"], columns=["target", "h"], values="relRMSE_AR")
    wide.columns = [f"{t}_h{h}" for t, h in wide.columns]
    for h in H_LIST: wide[f"mean_h{h}"] = wide[[f"{t}_h{h}" for t in TARGETS]].mean(axis=1)
    wide = wide.reset_index().sort_values("mean_h1").round(4)

    print("\n" + "="*104 + "\nPHASE 3  relRMSE vs AR(BIC)  (< 1 beats AR)\n" + "="*104)
    print(wide.to_string(index=False))

    print("\nJoint Model Confidence Set (90%) per target, h=1 [included]:")
    mcs_inc = {}
    try:
        from arch.bootstrap import MCS
        for t in TARGETS:
            a = actual[(t, 1)]
            L = pd.DataFrame({m: (a - fc[(m, t, 1)])**2 for m in models})
            mcs = MCS(L, size=0.10, reps=1000, block_size=4, method="R", seed=20260710); mcs.compute()
            mcs_inc[t] = sorted(mcs.included)
            print(f"  {t:7s} {len(mcs_inc[t])}/{L.shape[1]} models  "
                  f"excluded: {sorted(set(L.columns) - set(mcs_inc[t]))}")
    except Exception as ex:
        print(f"  MCS skipped: {type(ex).__name__}: {ex}")

    long.to_csv(OUT / "phase3_foundation_results.csv", index=False)
    wide.to_csv(OUT / "phase3_foundation_relrmse_ar.csv", index=False)
    fcs = pd.DataFrame([{"model": m, "target": t, "h": h, "origin": dates[i].date(),
                         "forecast": fc[(m, t, h)][k], "actual": actual[(t, h)][k]}
                        for (m, t, h) in fc for k, i in enumerate(origins)])
    fcs.to_csv(OUT / "phase3_forecasts.csv", index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = wide.set_index("model")[[f"{t}_h1" for t in TARGETS] + [f"{t}_h4" for t in TARGETS]]
    fig, ax = plt.subplots(figsize=(9, 0.42*len(M)+1.6))
    im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn_r", vmin=0.7, vmax=1.3)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([c.replace("_", "\n") for c in M.columns], fontsize=8)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=8)
    for (r, c), v in np.ndenumerate(M.values):
        if np.isfinite(v): ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.axvline(len(TARGETS)-0.5, color="k", lw=0.8)
    ax.set_title("Phase 3: relative RMSE vs AR(BIC)\nfoundation models vs VARs, h = 1 (left) and h = 4 (right)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.6, label="RMSE / AR(BIC)"); plt.tight_layout()
    plt.savefig(FIG / "phase3_heatmap.png", dpi=140, bbox_inches="tight"); plt.close()

    (OUT / "phase3_summary.json").write_text(json.dumps({
        "sample": f"{dates.min().date()} to {dates.max().date()}", "n_quarters": N,
        "oos_origins": f"{dates[origins[0]].date()} to {dates[origins[-1]].date()} ({len(origins)})",
        "best_mean_relRMSE_AR_h1": wide.iloc[0]["model"],
        "best_mean_relRMSE_AR_h4": wide.sort_values("mean_h4").iloc[0]["model"],
        "mcs90_h1": mcs_inc, "runtime_s": round(time.time()-t0)}, indent=2, default=str))
    print(f"\n[saved] outputs/phase3_*.csv, outputs/phase3_summary.json, outputs/figs/phase3_heatmap.png  "
          f"({time.time()-t0:.0f}s)\nDONE.")
