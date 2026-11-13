# Results (executed horse-race outputs)

Derived artifacts produced by `src/var_horserace.py` (one run per `--spec`),
`src/var_horserace_gap.py` and `src/phase2_modelclass.py` on **public FRED** data
(cached in `fred_data/`; `src/reproduce.py` fetches anything missing). Five specifications
are reported:

- base (`var_horserace_results.csv` / `var_horserace_summary.json`)
- all-YoY (`*_allyoy`), augmented (`*_augmented`), REER-YoY (`*_reer_yoy`)
- gap-vs-level (`var_horserace_gap_results.csv` / `_summary.json`)
- model-class phase 2 (`phase2_modelclass_results.csv` / `phase2_summary.json`)

Figures in `figs/` are the rel-RMSE heatmaps and winner-forecast charts for each spec.
Metrics: relative-RMSE vs benchmark with Clark-West, Diebold-Mariano, and Model Confidence Set.
