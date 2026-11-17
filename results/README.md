# Results (executed horse-race outputs)

Derived artifacts produced by `src/var_horserace.py` (one run per `--spec`),
`src/var_horserace_gap.py` and `src/phase2_modelclass.py` on **public FRED** data
(cached in `fred_data/`; `src/reproduce.py` fetches anything missing). Four specifications
are reported, with three further analyses:

- base (`var_horserace_results.csv` / `var_horserace_summary.json`)
- all-YoY (`*_allyoy`), augmented (`*_augmented`), REER-YoY (`*_reer_yoy`)
- gap-vs-level (`var_horserace_gap_results.csv` / `_summary.json`)
- model-class phase 2 (`phase2_modelclass_results.csv` / `phase2_summary.json`)
- independent cross-check of the winner plus extensions (`crosscheck/`, from `src/crosscheck_study1.py`)

Figures in `figs/` are the rel-RMSE heatmaps and winner-forecast charts for each spec.
Metrics: relative-RMSE vs benchmark with Clark-West, Diebold-Mariano, and Model Confidence Set.

The phase-2 files and `figs/phase2_heatmap.png` come from a run on 127 quarters (1994Q2 to 2025Q4).
`reproduce_results.json` uses 128 quarters (to 2026Q1), so its model-class figures differ from the
phase-2 table in the second or third decimal (Ridge small: 0.8792 there, 0.8768 here).
