# Macro VAR Selection Horse Race — Methodology

A defensible framework for selecting a baseline classical VAR, designed so it can survive scrutiny from a quant boss and later support a fair comparison against BVAR / Ridge / LASSO / ElasticNet / GVAR.

---

## 1. Executive summary

**Decide the objective first, because it changes everything.** Your comparison set (BVAR, shrinkage VARs, GVAR) is a *forecasting* lineup. So this is a **forecast horse race**, and structural identification (Cholesky ordering, sign restrictions) is **irrelevant to the selection** — it does not affect reduced-form point forecasts. Keep economic-sign sanity as a *secondary filter on the finalists*, not as a search axis.

Four design changes drive the rest of this document:

1. **Drop "residualisation against the endogenous block" as a search dimension.** If a variable is inside the VAR, the VAR already controls for the other variables and their lags through its own coefficients and reduced-form residual covariance. Pre-regressing a candidate on GDP/inflation/rate/FX and then putting it in a system that *also* contains those variables is redundant for forecasting and destroys any structural interpretation (you have orthogonalised twice). Residualisation is only legitimate as **external-instrument / narrative-shock construction** — a different exercise (see §2.5). This single fix removes most of your permutations.

2. **Search over economic *blocks*, not individual variables.** PFI, PFI-non-residential and gross fixed investment are three measures of one thing — they are collinear and must not co-enter. Pick **one representative per block** (investment, consumption, oil, fiscal). This turns a 2^N variable search into a 4-block search.

3. **Fix exogeneity a priori from economics, not by search.** For a small open economy, oil and foreign demand are plausibly exogenous → enter as **VARX exogenous regressors, un-residualised**. Investment/consumption/fiscal respond to the cycle → candidates for the *endogenous* block. You are not "testing" oil's exogeneity by brute force; you are imposing a defensible prior.

4. **Never average raw RMSE across targets** — GDP, inflation, rate and FX live on different scales. Use **RMSE relative to a per-target benchmark** (random walk and AR(p)), i.e. a Theil-U style ratio, then average *ranks* of those ratios.

The guard against the data-mining you (rightly) fear is the **Model Confidence Set** (Hansen–Lunde–Nason 2011) plus **Clark–West** tests for nested comparisons. That is the part that makes "I ran many models and this one won" defensible rather than lucky.

**Set expectations now:** the best classical VAR will most likely *not* win the eventual horse race against BVAR/shrinkage — that is the expected result in short macro samples, and the point of the classical VAR is to be the *interpretable benchmark* that quantifies how much shrinkage buys you. And FX probably will not beat a random walk at short horizons (Meese–Rogoff). Say this up front so the result is framed correctly.

---

## 2. Recommended methodology

A **staged pipeline with guardrails**, not a Cartesian grid.

### 2.0 Data prep & diagnostics (run once, not looped)
- Unit-root tests per series: ADF + KPSS + PP (use the pair — ADF and KPSS disagree informatively).
- **Cointegration on the core block**: Johansen trace/max-eigen. This matters because your framework currently only says "difference it," which silently rules out a VECM. Decide a *transformation strategy*: levels VAR (Sims–Stock–Watson: consistent under unit roots, avoids imposing wrong differencing), differences, or VECM if cointegration is strong. Fix this strategy; do **not** search over it and then compare RMSE across different target transforms — that is comparing apples to oranges.
- Define the OOS scheme once: expanding (preferred for macro) or rolling window, forecast horizons h ∈ {1, 4, 8} quarters, and the holdout span.
- Compute **benchmark forecasts**: per-target random walk and AR(p)-by-BIC. Everything is scored relative to these.

### 2.1 Anchor the core
Estimate the core VAR {GDP, inflation, policy rate, FX}. Select its lag order by IC **and** OOS performance (they can disagree; OOS wins for a forecasting model). This is your reference point.

### 2.2 One representative per block
Within each candidate block, choose the single best variable by data quality + univariate predictive content for the targets (e.g. simple marginal-R² or single-equation OOS gain). Carry forward ~4 block representatives, not ~15 collinear series.

### 2.3 Block-level forward selection (your "stepwise as diagnostic" compromise)
Start from core. Add one block representative at a time; **keep it only if** it improves average relative-RMSE *and* passes the validity filters (§7). Optionally one backward pass. With 4 blocks this is ~10 estimable models, not billions. This is exactly the "use stepwise to see what is consistently useful, not as final selection" you asked for — applied at the block level where it is statistically far safer.

### 2.4 Fine-tune only the survivors
For the handful of surviving specs, search VAR lag order p ∈ {1..4} and (only where economically motivated) the stationarity transform. **Critical hygiene:** lag/transform selection must happen *inside each training window* of the rolling/expanding scheme, never on the full sample — otherwise you leak the test set into selection.

### 2.5 Residualisation — the *only* defensible uses
If you genuinely want "shocks," do it properly, as a separate identification step on the chosen model — not as a pre-filter search:
- **Proxy-SVAR / external instruments** (Stock–Watson 2018; Mertens–Ravn 2013; Gertler–Karadi 2015): the shock series is an *instrument from outside the system*, not a residual stuffed back in.
- **Narrative cleaning of a policy variable** (Romer–Romer 2004 monetary, 2010 fiscal; Blanchard–Perotti 2002): regress the policy instrument on the *policymaker's information set* to strip endogenous response — appropriate for the policy rate or fiscal series specifically, not mechanically for every candidate.
- A cyclically-adjusted fiscal balance is itself a residualised series — fine to use as an input, but don't *also* residualise it again inside the VAR.

Everything else in your "residualisation choices (a)–(e) × lags" matrix should be dropped.

---

## 3. What to loop over

| Dimension | Range | Why it's safe |
|---|---|---|
| Block inclusion | core + forward selection over ~4 block representatives | small, theory-pruned, interpretable |
| VAR lag order p | 1–4 (extend to 6 only if T comfortably allows) | standard; selected inside training window |
| Forecast horizon h | {1, 4, 8} quarters | reporting dimension, not a selection trap |
| Stationarity strategy | a single fixed choice (levels / diff / VECM) decided in §2.0 | not searched against RMSE |
| OOS window type | one choice (expanding preferred); rolling as robustness | consistent across all models |

Total estimable specs: **dozens**, every one of which you can name and justify.

---

## 4. What NOT to loop over (valid vs dangerous)

- **Do not** loop over all 2^N variable subsets — multiple-testing disaster and infeasible. (Block forward selection instead.)
- **Do not** loop residualisation target × residualisation lag for in-system variables — redundant for forecasts, breaks structure (§1.1, §2.5).
- **Do not** co-enter collinear measures of one concept (PFI vs PFI-non-res vs GFI) — pick one (§2.2).
- **Do not** include Cholesky ordering or sign restrictions as a selection axis — **identification does not change reduced-form point forecasts**, only IRFs/variance decompositions. Looping it for a forecast race is wasted compute.
- **Do not** select lags/transforms/standardisation on the full sample, then evaluate OOS — that is leakage and inflates every RMSE.
- **Do not** mix target transforms across specs and then compare RMSE — fix the target's transform so the loss is on the same units.
- **Do not** average raw RMSE across targets — use relative-RMSE ranks (§1.4, §5).

---

## 5. Model ranking criteria

**Hard filters (pass/fail, applied before ranking):**
- Stability: all companion-matrix eigenvalues strictly inside the unit circle (max modulus < 1).
- Residual whiteness: portmanteau / LM autocorrelation test — flag severe violations (don't fetishise a single p-value cutoff in finite samples, but reject clear failures).
- Parsimony / degrees of freedom: total params = K(Kp + 1). For K=4, p=4 that is 68 params — already heavy on ~120 quarterly obs. Require a sane ratio (e.g. T / params ≥ ~3) or push anything larger to the shrinkage track.
- Conditioning: reject singular / ill-conditioned moment matrices (check condition number).

**Primary ranking (among survivors):**
- Average **rank of relative-RMSE** (vs RW/AR benchmark) across {GDP, inflation, rate, FX}, at the horizon(s) of interest. Use a weighted average if some targets matter more to the desk.

**Significance of forecast differences (the anti-data-mining layer):**
- **Diebold–Mariano** for non-nested pairwise comparisons; **Clark–West** for nested ones (OLS-VAR is nested in a BVAR over the same variables, and OLS is the α→0 limit of the penalised estimators — DM is *invalid* there).
- **Model Confidence Set** (Hansen–Lunde–Nason) across all candidates to report the set of models statistically indistinguishable from the best. If your "winner" is in a 12-model MCS, you say so honestly — that is what defensibility looks like.

**Tiebreak among MCS survivors:** parsimony first, then economic sign sanity of IRFs (the only place the structural overlay enters), then stability margin.

Secondary/reporting metrics: MAE, and directional accuracy with a **Pesaran–Timmermann** test for significance (raw hit-rates are misleading).

---

## 6. Pseudocode

```python
# ---- Stage 0: once ----
transforms = decide_transform_strategy(series, adf, kpss, pp, johansen)  # fixed, not searched
windows    = make_expanding_windows(T, min_train, horizons=[1,4,8])
benchmarks = {tgt: {"rw": rw_forecast(tgt, windows),
                    "ar": ar_bic_forecast(tgt, windows)} for tgt in TARGETS}

# ---- Stage 2.2: one representative per economic block ----
block_rep = {}
for block, members in CANDIDATE_BLOCKS.items():          # investment / consumption / oil / fiscal
    block_rep[block] = pick_best_by_univariate_gain(members, TARGETS, windows)
exog_blocks  = {"oil", "foreign_demand"}                 # a-priori exogenous -> VARX, NOT residualised
endo_blocks  = set(block_rep) - exog_blocks

# ---- Stages 2.1 + 2.3: anchor + block forward selection ----
def evaluate(spec):
    fold_errs = {tgt: [] for tgt in TARGETS}
    for tr, te in windows:                               # ALL selection inside the training fold
        p   = select_lag_order(spec, tr, ic="bic", oos=True)   # no peeking at te
        fit = fit_var_or_varx(spec, tr, p, transforms)
        if not passes_filters(fit, tr): return None      # stability / whiteness / dof / conditioning
        fc  = fit.forecast(te.horizons)
        for tgt in TARGETS: fold_errs[tgt].append(err(fc[tgt], te[tgt]))
    rel = {tgt: rmse(fold_errs[tgt]) / rmse(benchmarks[tgt]["rw"]) for tgt in TARGETS}  # Theil-U style
    return Result(spec, rel, params=n_params(fit), pmax_eig=max_modulus(fit))

results, current = [], Spec(endog=CORE)
results.append(evaluate(current))
improved = True
while improved:                                          # greedy forward at BLOCK level (~10 fits)
    improved = False
    for b in endo_blocks - current.used:
        cand = current.add_endog(block_rep[b])
        r = evaluate(cand)
        if r and avg_rank_gain(r, current.best): current, improved = cand, True; results.append(r)
    for b in exog_blocks - current.used:
        cand = current.add_exog(block_rep[b])            # VARX, no residualisation
        r = evaluate(cand)
        if r and avg_rank_gain(r, current.best): current, improved = cand, True; results.append(r)

# ---- Stage 2.4: fine lag tune on survivors only ----
finalists = [refit_over_p(r.spec, p_range=range(1,5), windows=windows) for r in top_k(results)]

# ---- Stage 5: rank + significance ----
ranked = rank_by_avg_relative_rmse(finalists)
for m in finalists: m.cw = clark_west(m, benchmark) ; m.dm = diebold_mariano(m, ranked[0])
mcs = arch_model_confidence_set([m.loss_series for m in finalists])   # arch.bootstrap.MCS
winner = parsimony_tiebreak([m for m in ranked if m in mcs])
```

Store one row per spec:

```
model_id | objective(forecast/struct) | endogenous_set | exogenous_set | target_transform |
var_lag_order | horizon | rmse_gdp | rmse_inf | rmse_rate | rmse_fx |
relRMSE_gdp | relRMSE_inf | relRMSE_rate | relRMSE_fx | avg_relRMSE_rank |
AIC | BIC | HQIC | n_params | T_over_params | max_eig_modulus | stability_pass |
whiteness_pvalue | cond_number | DM_vs_best_p | CW_vs_bench_p | in_MCS | comments
```
(Note your original list lacked the relative-RMSE, benchmark, horizon, target-transform, conditioning and MCS columns — those are the ones that make the table defensible.)

---

## 7. Final recommended implementation plan

1. **Stage 0 — diagnostics & scaffolding.** Unit roots, Johansen, fixed transform strategy, expanding-window splits, RW + AR(p) benchmarks. Tools: `statsmodels` (`VAR`, `VECM`, `adfuller`, `kpss`, `coint_johansen`).
2. **Stage 1 — anchor core VAR**, lag by BIC + OOS.
3. **Stage 2 — block representatives**, with a-priori exogenous/endogenous split fixed from economics.
4. **Stage 3 — block-level forward selection** with filters, all selection inside training folds.
5. **Stage 4 — fine lag tuning** on survivors.
6. **Stage 5 — rank** by average relative-RMSE; **Clark–West** vs benchmark, **DM** pairwise, **Model Confidence Set** (`arch.bootstrap.MCS` in Kevin Sheppard's `arch`); parsimony + sign-sanity tiebreak.
7. **Stage 6 — structural overlay (optional)** on the chosen model only: identify shocks via proxy-SVAR / narrative cleaning, check IRF signs. This is reporting, not selection.

### Phase 2 — fair comparison against BVAR / Ridge / LASSO / ElasticNet / GVAR
Hold these constant for *every* model: variable set, target transforms, train/test split, window scheme, horizons, metric, and benchmark.
- Tune all hyperparameters **on training data only**: BVAR Minnesota tightness λ via the marginal likelihood (Giannone–Lenza–Primiceri 2015); ridge/lasso/elastic-net penalties via **forward-chaining (blocked) time-series CV**, never random k-fold (leakage).
- Use **Clark–West, not DM, for nested comparisons** — OLS-VAR is nested in a same-variable BVAR (λ→∞ shrinks to prior) and is the α→0 limit of the penalised estimators.
- Score everything as relative-RMSE vs the same benchmark, then run **one MCS across the entire field** (classical + shrinkage together). That joint MCS is the genuinely fair "who wins."
- **GVAR is not apples-to-apples** unless you actually have the multi-country panel and weight matrices, and the target/information set is matched. If you don't, exclude it or report it separately with that caveat — don't let it contaminate the like-for-like comparison.

**Honest framing:** the value of this exercise is not "the classical VAR won." It is (a) a transparent, theory-pruned, leakage-free pipeline, (b) an interpretable classical benchmark, and (c) a statistically rigorous statement — via the MCS — of how much (if anything) shrinkage and Bayesian priors actually buy you in *your* sample.
