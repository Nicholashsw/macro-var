# Checking Study 1 in EViews

The commands below rebuild the CORE+OIL(x) VARX in EViews and give the numbers it
should return, so the Python pipeline can be checked against a second package. They
are single commands for the EViews command window, so they also work in EViews
Student Version Lite (no programs, 60 objects per page). Expected values come from
`src/crosscheck_study1.py` on the `fred_data/` vintage of 2026-08-21; pulling fresh
FRED data inside EViews will move the later decimals.

Type one line at a time. Lines starting with `'` are notes.

## 0. Workfile and data

```
wfcreate(wf=check, page=q) q 1947q1 2026q4
' Load the five CSVs from fred_data/: File > Import > Import from file, pick GDPC1.csv, accept the
' frequency conversion to quarterly by average. Repeat for PCEPILFE, FEDFUNDS, RBUSBIS,
' WTISPLC. (Or, if Wine has network access: dbopen(type=fred) fred  then
' fetch(d=fred, c=a) gdpc1 pcepilfe fedfunds rbusbis wtisplc   -- live vintage.)
smpl @all
series gdp = 400*dlog(gdpc1)
series inf = 400*dlog(pcepilfe)
series policy = fedfunds
series fx = 400*dlog(rbusbis)
series oil = 100*dlog(wtisplc)
series oilx = oil
```

Check: `smpl 1994q2 2026q1` then `show gdp inf policy fx oil`. Expect 128 rows with no NAs.

## 1. Full-sample VARX lag selection (sample 1994Q2-2026Q1)

```
smpl 1994q2 2026q1
var varx.ls 1 2 gdp inf policy fx @ c oil
varx.laglen(4)
```

Expected: **SC (Schwarz) picks lag 2.** AIC, FPE and HQ also pick 2. EViews scales the
criteria differently from statsmodels, so compare which lag gets the star, not the values.

## 2. Full-sample VARX(2) coefficients

```
var varx.ls 1 2 gdp inf policy fx @ c oil
```

Expected (first rows of each column):

| | GDP eq | INF eq | POLICY eq | FX eq |
|---|---|---|---|---|
| C | 2.9891 | 0.4386 | 0.0038 | -3.1778 |
| OIL | 0.1492 | 0.0252 | 0.0077 | -0.2413 |
| GDP(-1) | -0.1618 | -0.0119 | -0.0015 | 0.0846 |
| INF(-1) | -0.8120 | 0.5288 | -0.0409 | 2.2995 |

Optional diagnostics on the same VAR (compare p-values with `results/crosscheck/summary.json`):
`varx.arroots` (largest inverse root 0.8899), `varx.arlm(4)` (lag 1 p ≈ 0.034),
`varx.jbera` (joint p ≈ 0.000), `varx.white` (joint p ≈ 0.001),
`varx.testexog` (compare with `results/crosscheck/granger_block_exogeneity.csv`). LM, White and
normality use EViews' own small-sample variants, so expect the same verdicts with
slightly different statistics.

## 3. One expanding window: the first origin (estimate 1994Q2-2011Q4, forecast 2012Q1)

```
smpl 1994q2 2011q4
var v1.ls 1 2 gdp inf policy fx @ c oilx
v1.laglen(4)
' expected: SC picks lag 1
var v1.ls 1 1 gdp inf policy fx @ c oilx
' oil is unknown at the origin: use its AR(BIC) projection for 2012Q1 (AR(8), see check 3b)
smpl 2012q1 2012q1
oilx = 7.495022
v1.makemodel(m1)
m1.solve
show gdp_0 inf_0 policy_0 fx_0
```

Expected 2012Q1 forecasts:

| | GDP | INF | POLICY | FX |
|---|---|---|---|---|
| VARX CORE+OIL(x), lag 1 | 2.9975 | 1.3345 | 0.4891 | -2.5361 |
| VAR CORE, lag 1 (drop `@ c oilx`, use `@ c`) | 2.7209 | 1.2654 | 0.4428 | -1.1761 |
| AR(BIC) benchmark | 2.4849 | 1.6229 | 0.1631 | 2.5247 |
| Actual | 3.3404 | 2.5166 | 0.1033 | -2.9622 |

For the CORE row: `smpl 1994q2 2011q4`, `var v2.ls 1 1 gdp inf policy fx`,
`smpl 2012q1 2012q1`, `v2.makemodel(m2)`, `m2.solve`, `show gdp_0 inf_0 policy_0 fx_0`.

### 3b. Optional: the oil projection itself

BIC picks AR(8) for oil in this window. The sample shifts with the lag order:

```
smpl 1996q2 2011q4
equation e8.ls oil c oil(-1 to -8)
' one-step forecast for 2012Q1 should be 7.4950
smpl 2012q1 2012q1
e8.forecast oilf
show oilf
```

## 4. Unit roots and Johansen

```
smpl 1994q2 2026q1
gdp.uroot(adf, const, info=sic)
```
Expected ADF t: gdp -13.111, inf -2.993, policy -2.833, fx -8.737, oil -9.490
(lag lengths 0, 1, 1, 0, 0). PP Z(t): -13.19, -4.31, -1.92, -8.62, -9.40.

```
smpl 1994q1 2026q1
series lgdp = log(gdpc1)
series lp = log(pcepilfe)
series lfx = log(rbusbis)
group core_lv lgdp lp fedfunds lfx
core_lv.coint(c, 1 4)
```
Expected, case 3 (intercept in CE and VAR): trace 46.00, 26.68, 10.57, 3.51 against 5% critical
values 47.85, 29.80, 15.49, 3.84, so rank 0. This matches `results/reproduce_results.json` exactly.

## What cannot be checked by hand

The expanding-window loop (57 origins at h = 1, 54 at h = 4, each with its own lag
choice and oil projection) needs EViews' programming language, which Student Version
Lite does not have; its target numbers are in `results/crosscheck/relrmse.csv`. The Minnesota BVAR will
not match to the decimal either: the Python version puts a flat prior on the constant
and oil and scales by univariate AR(p) residuals, and EViews' defaults for those
terms differ.
