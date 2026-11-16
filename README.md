# Macro VAR

One question, asked two ways: what forecasts the US macroeconomy out of sample once
leakage, overfitting and multiple comparisons are controlled? The answer, across 1,000+
specifications, five estimator classes, growth-rate proxy and externally identified shocks, and a
real-time replication, is that nothing reliably beats a univariate autoregression.

Nicholas Hong, Nanyang Technological University

| Study | Paper | Headline |
|---|---|---|
| **Disciplined specification search and model-class evaluation** (repository root) | [`papers/01_var_selection_horserace.pdf`](papers/01_var_selection_horserace.pdf) | A theory-pruned block forward search with hard validity gates selects a four-variable core plus oil as the honest baseline; **no estimator — OLS, Minnesota BVAR, ridge, LASSO, elastic-net — is distinguishable from a univariate AR by the Model Confidence Set.** Full-sample HP filtering of the policy rate leaks the future by 66% of the cycle's own standard deviation at the endpoint; year-over-year transforms fail every whiteness test. |
| **A permutation study of VARX forecasting models** (`studies/varx_permutation_study/`) | [`studies/varx_permutation_study/paper/varx_permutation_study.pdf`](studies/varx_permutation_study/paper/varx_permutation_study.pdf) | 1,632 specification cells / 1,014 unique models in an expanding walk-forward from 2009Q4. Exogenous shocks buy a ~12% GDP RMSE gain at h=1 that is **mostly COVID and mostly noise** (4.0% of specs beat the core at the 5% level, what chance delivers); the gain survives ALFRED real-time vintages; **externally identified shocks (Kaenzig oil, Bauer-Swanson FOMC) do not beat cheap growth-rate proxies**; in the predictive density the shock model is overconfident until future-shock risk is priced, after which it is no sharper than the pure VAR. |

All data are public and keyless (FRED, FRBSF, NY Fed HLW, ALFRED); each study ships its results
tables and figures and can be re-executed.

![Relative RMSE heatmap](figs/relrmse_heatmap.png)

---

## Specification horse race (repository root)

```
src/            var_horserace.py (forward-selection runner; --spec base | augmented |
                reer_yoy | allyoy), var_horserace_gap.py (gap-vs-level), phase2_modelclass.py
                (OLS, BVAR, Ridge, LASSO, ElasticNet), figure builder, reproduce.py (single-seed
                driver that regenerates every number in the paper)
results/        executed outputs for all five specifications + phase 2 + reproduce_results.json
figs/           relative-RMSE heatmaps, winner forecasts, HP look-ahead, gap-vs-level
docs/           methodology note
papers/var_selection_horserace/   LaTeX source and figures
```

Every runner resolves its paths from its own file location, so they work from any working
directory (override the root with `REPRO_ROOT`). Re-runs write to `fred_data/`, `repro/` and
`outputs/` — all gitignored — so nothing regenerated ever overwrites the committed tables in
`results/`, the charts in `figs/`, or the typeset figures in `papers/var_selection_horserace/figs/`.
`src/reproduce.py` puts its three paper figures in `outputs/figs/`.

Re-run everything:

```
pip install -r requirements.txt
python src/reproduce.py            # pulls FRED + FRBSF TFP if not cached in fred_data/, seed 20260710
```

**Which number is the result.** The greedy forward path, *before* the validity gates are applied,
runs 1.0431 (`CORE`) → 0.9452 (`CORE+WEALTH`) → 0.9117 (`CORE+WEALTH+OIL(x)`) in mean relative RMSE
at h=1. Neither wealth specification is admissible: `CORE+WEALTH` fails the parsimony gate
(T/k = 2.29) and `CORE+WEALTH+OIL(x)` fails both parsimony (T/k = 2.10) and residual whiteness
(p = 0.044). The **eligible winner — the specification used throughout the paper — is `CORE+OIL(x)`
at 0.9679 (h=1) and 0.8786 (h=4)**. The rejected 0.9117 is reported precisely *because* it is the
best raw score in the run: it is the case that demonstrates the hard filters overruling
out-of-sample RMSE, which is the point of this study. Full per-specification detail is in
`results/reproduce_results.json` (`phaseA_table`).

Re-executed on 2026-08-21 against a fresh FRED pull: the specification search (eligible winner
`CORE+OIL(x)`, and the same greedy path and gate rejections above) and the model-class table
reproduce identically; the HP
revision statistic (66.3% vs 65.8%) and the Johansen trace move in the second decimal because the
reference run used a frozen 2026-07-10 FRED vintage and later vintages carry data revisions. `results/reproduce_results.json`
records the library versions of the reference run (numpy 2.4.4, pandas 3.0.2, statsmodels 0.14.6,
scikit-learn 1.8.0, arch 8.0.0); those exact pins are in `requirements-reference.txt`, so
`pip install -r requirements-reference.txt` rebuilds the reference environment. `requirements.txt`
is the loose everyday set.

### Cross-check

`src/crosscheck_study1.py` is a separate implementation of the eligible winner and the
CORE anchor, extended with a Minnesota BVAR in the same loop, unit
roots, Johansen under three trend cases, full residual diagnostics, IRFs, FEVD,
historical decomposition and a GLP-style BVAR. Outputs are in `results/crosscheck/`.

- Scored on the paper's window it returns the published numbers to four decimals:
  0.9679 / 0.8786 for `CORE+OIL(x)` and 1.0431 / 0.9318 for `CORE`, with identical
  Clark-West statistics. `reproduce.py` scores both horizons on the 54 common origins;
  `reproduce_results.json` used to report 57 for h=1 and now reports 54. On all 57 h=1
  origins the winner scores 0.9656.
- The Minnesota BVAR (lambda 0.1, prior mean 1 on the policy rate's own lag) averages
  0.901 at h=1 against the OLS VARX's 0.966, mostly from GDP and the policy rate, but no target
  beats the OLS VARX in a Diebold-Mariano test (p from 0.18 to 0.81). It is worse on
  inflation.
- The full-sample VARX(2) is stable (largest root 0.89) but rejects normality and
  White homoskedasticity, mostly from 2020, and shows lag-1 serial correlation
  (LM p = 0.034). Johansen finds no cointegration with an intercept (case 3) and one
  vector with no deterministic terms, so the no-VECM choice depends on the trend case.

`docs/eviews_spotchecks.md` lists EViews commands, with expected values, for checking the
same model in a second package.

![Cross-check one-step forecasts](figs/crosscheck_oos_h1.png)

## VARX permutation study

`studies/varx_permutation_study/README.md`. Self-contained: `fetch_data.py` then `run_study.py`
reproduces the full walk-forward in well under a minute; the extensions (real-time, identified
shocks, combination, density, Bayesian coverage) run in seconds against cached data. A
self-contained interactive console is in `dashboard/dashboard.html`. Re-executed end-to-end on
2026-08-21 (1,014 models, 38 s; all 31 paper statistics identical) and again on 2026-08-23
(1,014 models, 63,918 result rows, 36 s).

## What the two studies agree on

1. Parsimony and leakage-free evaluation dominate estimator sophistication in short macro samples.
2. Reduced-form forecasting does not reward structural identification; cheap proxies are the right
   instrument for that job.
3. Honest nulls are the product. Every number here was reported as it came out.

---

Nicholas Hong | Built for educational and research purposes. Not financial advice.
