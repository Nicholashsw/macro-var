#!/usr/bin/env python3
"""
crosscheck_study1.py
Separate implementation of the Study 1 winner, used to check reproduce.py,
plus a set of extensions.

Part 1 rebuilds the eligible winner CORE+OIL(x) and the CORE anchor:
  - FRED GDPC1, PCEPILFE, FEDFUNDS, RBUSBIS, WTISPLC; monthly -> quarterly mean
  - 400*dlog growth, fed funds in levels, 100*dlog oil; sample 1994Q2-2026Q1
  - expanding window from 55% of the sample, h = 1 and 4, SC lag (max 4)
    chosen inside every window, oil projected out of sample by AR(BIC)
  - RMSE relative to RW and AR(BIC), Clark-West against the RW
Part 2 extends it:
  - Minnesota BVAR on the CORE+OIL(x) specification in the same loop, with a
    Diebold-Mariano test against the OLS VARX
  - ADF / KPSS / Phillips-Perron, Johansen trace under three trend cases
  - full-sample VARX(SC) diagnostics: lag criteria, inverse roots, LM serial
    correlation, portmanteau, normality, White, Granger block exogeneity
  - Cholesky and generalized IRFs (bootstrap bands), FEVD, historical
    decomposition, and a GLP-style hierarchical BVAR
  - the values used by docs/eviews_spotchecks.md

    python src/crosscheck_study1.py

Reads fred_data/ (fetches anything missing, same endpoint as reproduce.py) and
writes to outputs/crosscheck/. The committed copy lives in results/crosscheck/.
"""
import json
import os
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import scipy
from scipy import stats
from scipy.special import multigammaln
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.simplefilter("ignore", InterpolationWarning)   # KPSS p-values are bounded; reported as such

ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "fred_data"
DATA.mkdir(exist_ok=True)
OUT = ROOT / "outputs" / "crosscheck"
OUT.mkdir(parents=True, exist_ok=True)

START, END = "1994-04-01", "2026-01-01"
MAXLAG_VAR, MAXLAG_AR, OOS_FRAC = 4, 8, 0.55
BVAR_L1, BVAR_L2, BVAR_L3 = 0.1, 0.99, 1.0
TARGETS = ["gdp", "inf", "policy", "fx"]
PRIOR_MEAN = [0.0, 0.0, 1.0, 0.0]      # own first lag: 1 for the level rate
SEED = 20260710
# paper numbers, from the committed reproduce.py run (frozen 2026-07-10 vintage)
_rep = json.loads((ROOT / "results" / "reproduce_results.json").read_text())
_tab = {r["spec"]: r for r in _rep["phaseA_table"]}
REF = {(h, m): [_tab[spec][f"{t}_h{h}"] for t in ("GDP", "INF", "POLICY", "FX")] + [_tab[spec][f"mean_h{h}"]]
       for h in (1, 4) for m, spec in (("VARX_CORE+OIL", "CORE+OIL(x)"), ("VAR_CORE", "CORE"))}


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_q(sid):
    f = DATA / f"{sid}.csv"
    if not f.exists():
        subprocess.run(["curl", "-sS", "--max-time", "60", "-o", str(f),
                        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"], check=True)
    df = pd.read_csv(DATA / f"{sid}.csv")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index().resample("QS").mean()


lv = {k: load_q(k) for k in ["GDPC1", "PCEPILFE", "FEDFUNDS", "RBUSBIS", "WTISPLC"]}
full = pd.DataFrame({
    "gdp": 400 * np.log(lv["GDPC1"]).diff(),
    "inf": 400 * np.log(lv["PCEPILFE"]).diff(),
    "policy": lv["FEDFUNDS"],
    "fx": 400 * np.log(lv["RBUSBIS"]).diff(),
    "oil": 100 * np.log(lv["WTISPLC"]).diff(),
})
panel = full.loc[START:END].copy()
assert panel.notna().all().all(), "missing values inside the sample"
N = len(panel)
I0 = int(np.floor(N * OOS_FRAC))
dates = panel.index
qlab = lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}"
print(f"Panel {qlab(dates[0])} to {qlab(dates[-1])}: {N} quarters, "
      f"first origin {qlab(dates[I0])}")


# --------------------------------------------------------------------------
# 2. Horse race
# --------------------------------------------------------------------------
def ar_bic(y, h, maxlag=MAXLAG_AR):
    """AR(p) by BIC, OLS, sample shifts with p; returns (path, p)."""
    y = np.asarray(y, float); n = len(y)
    best = (np.inf, 1, None)
    for p in range(1, maxlag + 1):
        if n - p < p + 5:
            break
        Y = y[p:]
        X = np.column_stack([np.ones(n - p)] + [y[p - 1 - j:n - 1 - j] for j in range(p)])
        b, *_ = np.linalg.lstsq(X, Y, rcond=None)
        r = Y - X @ b
        s2 = max(r @ r / len(Y), 1e-12)
        bic = len(Y) * np.log(s2) + (p + 1) * np.log(len(Y))
        if bic < best[0]:
            best = (bic, p, b)
    _, p, b = best
    hist = list(y[-p:]); out = []
    for _ in range(h):
        f = b[0] + sum(b[1 + j] * hist[-1 - j] for j in range(p))
        out.append(f); hist.append(f)
    return np.array(out), p


def fit_var(endog, exog=None):
    m = VAR(endog, exog=exog)
    res = m.fit(maxlags=MAXLAG_VAR, ic="bic", trend="c")
    if res.k_ar == 0:
        res = m.fit(1, trend="c")
    return res


def bvar_minnesota(endog, exog, p, h, exog_future):
    """Litterman/Minnesota BVAR with fixed diagonal Sigma from univariate AR(p)
    residual variances and a flat prior on the constant and oil. The own first
    lag has prior mean 1 for the level series (policy) and 0 for the growth
    rates, the same convention as phase2_modelclass.py.
    Returns the h-step posterior-mean forecast path."""
    Y_all = np.asarray(endog, float); E_all = np.asarray(exog, float).reshape(len(Y_all), -1)
    T, n = Y_all.shape
    Y = Y_all[p:]
    lags = [Y_all[p - l:T - l] for l in range(1, p + 1)]
    X = np.column_stack([np.ones(T - p), E_all[p:]] + lags)
    ndet = 1 + E_all.shape[1]
    sig = np.empty(n)
    for i in range(n):
        yi = Y_all[:, i]
        Xi = np.column_stack([np.ones(T - p)] + [yi[p - l:T - l] for l in range(1, p + 1)])
        b, *_ = np.linalg.lstsq(Xi, yi[p:], rcond=None)
        r = yi[p:] - Xi @ b
        sig[i] = np.sqrt(r @ r / (len(r) - Xi.shape[1]))
    B = np.zeros((X.shape[1], n))
    for i in range(n):
        prec = np.zeros(X.shape[1])            # prior precision; 0 = flat
        for l in range(1, p + 1):
            for j in range(n):
                sd = BVAR_L1 / l ** BVAR_L3 if i == j else \
                    BVAR_L1 * BVAR_L2 * sig[i] / (l ** BVAR_L3 * sig[j])
                prec[ndet + (l - 1) * n + j] = 1.0 / sd ** 2
        mean = np.zeros(X.shape[1]); mean[ndet + i] = PRIOR_MEAN[i]
        A = np.diag(prec) + X.T @ X / sig[i] ** 2
        B[:, i] = np.linalg.solve(A, prec * mean + X.T @ Y[:, i] / sig[i] ** 2)
    hist = list(Y_all[-p:]); out = []
    ef = np.asarray(exog_future, float).reshape(h, -1)
    for s in range(h):
        x = np.concatenate([[1.0], ef[s]] + [hist[-l] for l in range(1, p + 1)])
        f = x @ B; out.append(f); hist.append(f)
    return np.array(out)


MODELS = ["VARX_CORE+OIL", "VAR_CORE", "BVAR_CORE+OIL"]
fc, lagrec, spot = {}, {}, {}
for h in (1, 4):
    rows = []
    for i in range(I0, N - h):
        tr = panel.iloc[: i + 1]
        row = {"origin": qlab(dates[i]), "target_date": qlab(dates[i + h])}
        for t in TARGETS:
            row[f"{t}_actual"] = panel[t].iloc[i + h]
            row[f"{t}_RW"] = tr[t].iloc[-1]
            row[f"{t}_AR"] = ar_bic(tr[t].values, h)[0][-1]
        oil_path, oil_p = ar_bic(tr["oil"].values, h)
        rx = fit_var(tr[TARGETS].values, tr[["oil"]].values)
        fx_ = rx.forecast(tr[TARGETS].values[-rx.k_ar:], steps=h, exog_future=oil_path.reshape(h, 1))
        rc = fit_var(tr[TARGETS].values)
        fc_ = rc.forecast(tr[TARGETS].values[-rc.k_ar:], steps=h)
        fb_ = bvar_minnesota(tr[TARGETS].values, tr[["oil"]].values, rx.k_ar, h, oil_path)
        for k, t in enumerate(TARGETS):
            row[f"{t}_VARX_CORE+OIL"] = fx_[h - 1, k]
            row[f"{t}_VAR_CORE"] = fc_[h - 1, k]
            row[f"{t}_BVAR_CORE+OIL"] = fb_[h - 1, k]
        row["lag_varx"], row["lag_core"], row["lag_oil_ar"] = rx.k_ar, rc.k_ar, oil_p
        rows.append(row)
        if h == 1 and i == I0:
            spot["first_window"] = {
                "estimation_sample": f"{qlab(dates[0])} {qlab(dates[i])}",
                "forecast_date": qlab(dates[i + 1]),
                "varx_sc_lag": int(rx.k_ar), "core_sc_lag": int(rc.k_ar),
                "oil_ar_bic_lag": int(oil_p), "oil_projection": float(oil_path[0]),
                "varx_forecast": {t: float(fx_[0, k]) for k, t in enumerate(TARGETS)},
                "core_forecast": {t: float(fc_[0, k]) for k, t in enumerate(TARGETS)},
                "bvar_forecast": {t: float(fb_[0, k]) for k, t in enumerate(TARGETS)},
                "ar_bic_forecast": {t: float(row[f"{t}_AR"]) for t in TARGETS},
                "actual": {t: float(row[f"{t}_actual"]) for t in TARGETS},
            }
    fc[h] = pd.DataFrame(rows)
    fc[h].to_csv(OUT / f"oos_forecasts_h{h}.csv", index=False)
    lagrec[h] = fc[h][["lag_varx", "lag_core", "lag_oil_ar"]].value_counts().to_dict()


# --------------------------------------------------------------------------
# 3. Scoring
# --------------------------------------------------------------------------
def clark_west(a, f_small, f_large, h):
    es, el = a - f_small, a - f_large
    fh = es ** 2 - (el ** 2 - (f_small - f_large) ** 2)
    n = len(fh); d = fh - fh.mean(); v = d @ d / n
    for L in range(1, h):
        v += 2 * (d[L:] @ d[:-L]) / n
    v /= n
    cw = fh.mean() / np.sqrt(v)
    return float(cw), float(1 - stats.norm.cdf(cw))


def dm_test(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; n = len(d); dd = d - d.mean(); v = dd @ dd / n
    for L in range(1, h):
        v += 2 * (dd[L:] @ dd[:-L]) / n
    v /= n
    dm = d.mean() / np.sqrt(v) * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


rel_rows, cw_rows, dm_rows = [], [], []
for h in (1, 4):
    f = fc[h]
    for m in MODELS:
        rr, ra = {}, {}
        for t in TARGETS:
            a = f[f"{t}_actual"].values
            e = a - f[f"{t}_{m}"].values
            rm = np.sqrt((e ** 2).mean())
            rr[t] = rm / np.sqrt(((a - f[f"{t}_RW"].values) ** 2).mean())
            ra[t] = rm / np.sqrt(((a - f[f"{t}_AR"].values) ** 2).mean())
            cw, p = clark_west(a, f[f"{t}_RW"].values, f[f"{t}_{m}"].values, h)
            cw_rows.append({"h": h, "model": m, "target": t, "cw": cw, "p_one_sided": p})
        ref = REF.get((h, m))
        rel_rows.append({"h": h, "model": m, **{t: rr[t] for t in TARGETS},
                         "mean_vs_rw": np.mean(list(rr.values())),
                         "mean_vs_ar": np.mean(list(ra.values())), "origins": len(f),
                         **({f"paper_{t}": ref[k] for k, t in enumerate(TARGETS)} | {"paper_mean": ref[4]}
                            if ref else {})})
    for t in TARGETS:
        a = f[f"{t}_actual"].values
        dm, p = dm_test(a - f[f"{t}_BVAR_CORE+OIL"].values, a - f[f"{t}_VARX_CORE+OIL"].values, h)
        dm_rows.append({"h": h, "target": t, "dm_bvar_minus_ols": dm, "p_two_sided": p})

# reproduce.py scores both horizons on the common origins range(i0, N - max(H)),
# i.e. 54 for h = 1 as well. Score h = 1 on that window too, so the paper's
# numbers can be compared like for like.
f54 = fc[1].iloc[: N - 4 - I0]
paper_rows = []
for m in MODELS[:2]:
    rr = {}
    for t in TARGETS:
        a = f54[f"{t}_actual"].values
        rr[t] = np.sqrt(((a - f54[f"{t}_{m}"].values) ** 2).mean()) / \
            np.sqrt(((a - f54[f"{t}_RW"].values) ** 2).mean())
        cw, p = clark_west(a, f54[f"{t}_RW"].values, f54[f"{t}_{m}"].values, 1)
        rr[f"cw_{t}"], rr[f"cw_p_{t}"] = cw, p
    paper_rows.append({"h": 1, "model": m, "origins": len(f54), **rr,
                       "mean_vs_rw": np.mean([rr[t] for t in TARGETS]), "paper_mean": REF[(1, m)][4]})
pd.DataFrame(paper_rows).round(4).to_csv(OUT / "relrmse_h1_paper_window.csv", index=False)

rel = pd.DataFrame(rel_rows); cwt = pd.DataFrame(cw_rows); dmt = pd.DataFrame(dm_rows)
rel.round(4).to_csv(OUT / "relrmse.csv", index=False)
cwt.round(4).to_csv(OUT / "clark_west.csv", index=False)
dmt.round(4).to_csv(OUT / "bvar_vs_ols_dm.csv", index=False)


# --------------------------------------------------------------------------
# 4. Full-sample diagnostics
# --------------------------------------------------------------------------
def pp_test(x):
    """Phillips-Perron Z(t), constant, Bartlett kernel, Newey-West bandwidth."""
    x = np.asarray(x, float); y, ylag = x[1:], x[:-1]; T = len(y)
    X = np.column_stack([np.ones(T), ylag])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); u = y - X @ b
    s2 = u @ u / (T - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    t_rho = (b[1] - 1) / se
    g0 = u @ u / T
    L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lam2 = g0 + 2 * sum((1 - j / (L + 1)) * (u[j:] @ u[:-j]) / T for j in range(1, L + 1))
    zt = np.sqrt(g0 / lam2) * t_rho - (lam2 - g0) * T * se / (2 * np.sqrt(lam2) * np.sqrt(s2))
    return float(zt), float(mackinnonp(zt, regression="c")), L


ur = []
for s in ["gdp", "inf", "policy", "fx", "oil"]:
    x = panel[s].values
    a = adfuller(x, regression="c", autolag="BIC")
    with np.errstate(all="ignore"):
        k = kpss(x, regression="c", nlags="auto")
    zt, pp_p, bw = pp_test(x)
    ur.append({"series": s, "adf_t": a[0], "adf_p": a[1], "adf_lags": a[2],
               "kpss_lm": k[0], "kpss_p_bounded": k[1], "kpss_bw": k[2],
               "pp_zt": zt, "pp_p": pp_p, "pp_bw": bw})
urt = pd.DataFrame(ur); urt.round(4).to_csv(OUT / "unit_roots.csv", index=False)

core_lv = pd.DataFrame({"lgdp": np.log(lv["GDPC1"]), "lp": np.log(lv["PCEPILFE"]),
                        "fedfunds": lv["FEDFUNDS"], "lfx": np.log(lv["RBUSBIS"])}
                       ).dropna().loc["1994-01-01":END]
jo = []
for det, case in ((-1, "no deterministic terms (EViews case 1)"),
                  (0, "constant (EViews case 3)"),
                  (1, "constant + linear trend (approx. EViews case 5)")):
    j = coint_johansen(core_lv.values, det_order=det, k_ar_diff=4)
    jo.append({"det_order": det, "case": case,
               "trace": np.round(j.lr1, 2).tolist(), "crit5": np.round(j.cvt[:, 1], 2).tolist(),
               "rank_5pct": int(np.argmin(np.append(j.lr1 > j.cvt[:, 1], False)))})
pd.DataFrame(jo).to_csv(OUT / "johansen.csv", index=False)

Y, E = panel[TARGETS], panel[["oil"]]
vm = VAR(Y, exog=E)
sel = vm.select_order(MAXLAG_VAR, trend="c")
crit = pd.DataFrame({ic: [sel.ics[ic][p] for p in range(MAXLAG_VAR + 1)]
                     for ic in ("aic", "bic", "fpe", "hqic")})
crit.index.name = "lag"
crit.round(4).to_csv(OUT / "lag_criteria.csv")
pf = max(int(sel.selected_orders["bic"]), 1)
res = vm.fit(pf, trend="c")
n, T = len(TARGETS), res.nobs
U = res.resid.values
Xreg = np.column_stack([np.ones(T), E.values[pf:]] +
                       [Y.values[pf - l:len(Y) - l] for l in range(1, pf + 1)])

comp = np.zeros((n * pf, n * pf))
comp[:n, :] = np.hstack([res.coefs[l] for l in range(pf)])
if pf > 1:
    comp[n:, :-n] = np.eye(n * (pf - 1))
inv_roots = np.sort(np.abs(np.linalg.eigvals(comp)))[::-1]


def lm_serial(U, X, h, only_lag_h=False):
    T, n = U.shape
    lags = [h] if only_lag_h else range(1, h + 1)
    Ul = np.column_stack([np.vstack([np.zeros((L, n)), U[:-L]]) for L in lags])
    Z = np.column_stack([X, Ul])
    B, *_ = np.linalg.lstsq(Z, U, rcond=None)
    Se = (U - Z @ B).T @ (U - Z @ B) / T
    Su = U.T @ U / T
    lm = T * (n - np.trace(np.linalg.solve(Su, Se)))
    df = len(list(lags)) * n * n
    return float(lm), int(df), float(stats.chi2.sf(lm, df))


lm_rows = []
for hh in range(1, 5):
    a, dfa, pa = lm_serial(U, Xreg, hh, True)
    b, dfb, pb = lm_serial(U, Xreg, hh, False)
    lm_rows.append({"lag": hh, "lm_lag_h": a, "df_h": dfa, "p_lag_h": pa,
                    "lm_lags_1_to_h": b, "df_1h": dfb, "p_lags_1_to_h": pb})

wt = res.test_whiteness(nlags=8, adjusted=True)
nt = res.test_normality()

Xw = np.column_stack([np.ones(T), Xreg[:, 1:], Xreg[:, 1:] ** 2])
prods, names = [], []
for a_ in range(n):
    for b_ in range(a_, n):
        prods.append(U[:, a_] * U[:, b_]); names.append(f"{TARGETS[a_]}*{TARGETS[b_]}")
V = np.column_stack(prods)
Bw, *_ = np.linalg.lstsq(Xw, V, rcond=None)
Ru = V - Xw @ Bw; Rr = V - V.mean(0)
m_ = V.shape[1]; q_ = Xw.shape[1] - 1
white_lm = T * (m_ - np.trace(np.linalg.solve(Rr.T @ Rr / T, Ru.T @ Ru / T)))
white_df = m_ * q_
white_comp = [{"component": nm, "r2": 1 - Ru[:, k] @ Ru[:, k] / (Rr[:, k] @ Rr[:, k]),
               "chi2": T * (1 - Ru[:, k] @ Ru[:, k] / (Rr[:, k] @ Rr[:, k])),
               "p": float(stats.chi2.sf(T * (1 - Ru[:, k] @ Ru[:, k] / (Rr[:, k] @ Rr[:, k])), q_))}
              for k, nm in enumerate(names)]

gr = []
for dep in TARGETS:
    others = [v for v in TARGETS if v != dep]
    for c in others + ["ALL"]:
        cs = others if c == "ALL" else [c]
        tc = res.test_causality(dep, cs, kind="wald")
        gr.append({"dependent": dep, "excluded": c, "chi2": float(tc.test_statistic),
                   "df": int(tc.df), "p": float(tc.pvalue)})
grt = pd.DataFrame(gr); grt.round(4).to_csv(OUT / "granger_block_exogeneity.csv", index=False)

H_IRF = 12
irf = res.irf(H_IRF)
oirf, oirf_se = irf.orth_irfs, irf.stderr(orth=True)
Sig = res.sigma_u.values
Phi = res.ma_rep(H_IRF)


def girf(Phi, Sig):
    return np.stack([Phi[s] @ Sig / np.sqrt(np.diag(Sig)) for s in range(len(Phi))])


G = girf(Phi, Sig)
rng = np.random.default_rng(SEED)
B_REP = 500
Gb = np.empty((B_REP,) + G.shape)
yv, ev = Y.values, E.values
intercept, exog_coef = res.params.values[0], res.params.values[1]
for r in range(B_REP):
    ub = U[rng.integers(0, T, T)]
    ys = yv.copy()
    for t in range(pf, len(yv)):
        ys[t] = intercept + exog_coef * ev[t, 0] + sum(res.coefs[l] @ ys[t - 1 - l] for l in range(pf)) + ub[t - pf]
    rb = VAR(ys, exog=ev).fit(pf, trend="c")
    Gb[r] = girf(rb.ma_rep(H_IRF), rb.sigma_u)
G_se = Gb.std(0)

irf_rows = []
for s in range(H_IRF + 1):
    for i, resp in enumerate(TARGETS):
        for j, imp in enumerate(TARGETS):
            irf_rows.append({"horizon": s, "response": resp, "impulse": imp,
                             "chol": oirf[s, i, j], "chol_se": oirf_se[s, i, j],
                             "gen": G[s, i, j], "gen_se_boot": G_se[s, i, j]})
pd.DataFrame(irf_rows).round(5).to_csv(OUT / "irf.csv", index=False)

fe = res.fevd(H_IRF)
fevd_rows = [{"variable": v, "horizon": s + 1, **{f"share_{TARGETS[j]}": 100 * fe.decomp[i][s, j]
                                                  for j in range(n)}}
             for i, v in enumerate(TARGETS) for s in range(H_IRF)]
pd.DataFrame(fevd_rows).round(3).to_csv(OUT / "fevd_cholesky.csv", index=False)

P = np.linalg.cholesky(Sig)
eps = U @ np.linalg.inv(P).T
Theta = [m @ P for m in res.ma_rep(T)]
contrib = np.zeros((T, n, n))
for t in range(T):
    for s in range(t + 1):
        contrib[t] += Theta[s] * eps[t - s][None, :]
hd = pd.DataFrame(index=Y.index[pf:])
for i, v in enumerate(TARGETS):
    hd[f"{v}_actual"] = Y[v].values[pf:]
    for j, sh in enumerate(TARGETS):
        hd[f"{v}_from_{sh}_shock"] = contrib[:, i, j]
    hd[f"{v}_base_and_oil"] = Y[v].values[pf:] - contrib[:, i, :].sum(1)
hd.round(4).to_csv(OUT / "historical_decomposition.csv")


def glp_bvar(Yd, Ed, p):
    """Conjugate Minnesota (NIW) BVAR with dummy observations, prior mean 0,
    lambda chosen by maximising log marginal likelihood + Gamma hyperprior
    (mode 0.2, sd 0.4) as in Giannone, Lenza and Primiceri (2015); scale psi
    fixed at AR(1) residual variances."""
    Yd = np.asarray(Yd, float); Ed = np.asarray(Ed, float).reshape(len(Yd), -1)
    T_, n_ = Yd.shape
    Yt = Yd[p:]
    Xt = np.column_stack([Yd[p - l:T_ - l] for l in range(1, p + 1)] + [np.ones(T_ - p), Ed[p:]])
    k_ = Xt.shape[1]; ndet = k_ - n_ * p
    psi = np.empty(n_)
    for i in range(n_):
        X1 = np.column_stack([np.ones(T_ - 1), Yd[:-1, i]])
        b, *_ = np.linalg.lstsq(X1, Yd[1:, i], rcond=None)
        r = Yd[1:, i] - X1 @ b
        psi[i] = np.sqrt(r @ r / (T_ - 3))

    def dummies(lam):
        Yd1 = np.zeros((n_ * p, n_)); Xd1 = np.zeros((n_ * p, k_))
        Xd1[:, :n_ * p] = np.kron(np.diag(np.arange(1, p + 1)), np.diag(psi)) / lam
        Yd2 = np.diag(psi); Xd2 = np.zeros((n_, k_))
        Yd3 = np.zeros((ndet, n_)); Xd3 = np.zeros((ndet, k_)); Xd3[:, n_ * p:] = 1e-4 * np.eye(ndet)
        return np.vstack([Yd1, Yd2, Yd3]), np.vstack([Xd1, Xd2, Xd3])

    def logml(Yx, Xx):
        Tx = Yx.shape[0]
        XtX = Xx.T @ Xx
        B = np.linalg.solve(XtX, Xx.T @ Yx); S = (Yx - Xx @ B).T @ (Yx - Xx @ B)
        return (-n_ / 2 * np.linalg.slogdet(XtX)[1] - (Tx - k_) / 2 * np.linalg.slogdet(S)[1]
                + multigammaln((Tx - k_) / 2, n_) - n_ * Tx / 2 * np.log(np.pi))

    sd_, mode_ = 0.4, 0.2
    b_rate = (mode_ + np.sqrt(mode_ ** 2 + 4 * sd_ ** 2)) / (2 * sd_ ** 2)
    a_shape = 1 + mode_ * b_rate
    grid = np.exp(np.linspace(np.log(0.01), np.log(5), 400))
    lp = []
    for lam in grid:
        Ydum, Xdum = dummies(lam)
        ml = logml(np.vstack([Yt, Ydum]), np.vstack([Xt, Xdum])) - logml(Ydum, Xdum)
        lp.append(ml + stats.gamma.logpdf(lam, a_shape, scale=1 / b_rate))
    lp = np.array(lp); lam_star = grid[lp.argmax()]
    Ydum, Xdum = dummies(lam_star)
    Ys, Xs = np.vstack([Yt, Ydum]), np.vstack([Xt, Xdum])
    B = np.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)
    Bols, *_ = np.linalg.lstsq(Xt, Yt, rcond=None)
    return {"lambda_star": float(lam_star), "log_posterior_max": float(lp.max()),
            "lambda_grid_edge": bool(lp.argmax() in (0, len(grid) - 1)),
            "own_lag1_glp": {TARGETS[i]: float(B[i, i]) for i in range(n_)},
            "own_lag1_ols": {TARGETS[i]: float(Bols[i, i]) for i in range(n_)},
            "oil_coef_glp": {TARGETS[i]: float(B[-1, i]) for i in range(n_)},
            "oil_coef_ols": {TARGETS[i]: float(Bols[-1, i]) for i in range(n_)}}


glp = glp_bvar(Y.values, E.values, pf)


# --------------------------------------------------------------------------
# 5. Figures, summary, values for docs/eviews_spotchecks.md
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f1 = fc[1]
x = pd.PeriodIndex(f1["target_date"].str.replace("Q", "-Q"), freq="Q").to_timestamp()
fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for ax, t in zip(axs.flat, TARGETS):
    ax.plot(x, f1[f"{t}_actual"], color="black", lw=1.6, label="Actual")
    ax.plot(x, f1[f"{t}_VARX_CORE+OIL"], label="VARX CORE+OIL(x)")
    ax.plot(x, f1[f"{t}_BVAR_CORE+OIL"], label="BVAR Minnesota")
    ax.plot(x, f1[f"{t}_AR"], label="AR(BIC)", lw=0.9)
    ax.plot(x, f1[f"{t}_RW"], label="Random walk", lw=0.9, ls="--")
    ax.set_title(f"{t}: one-step-ahead forecasts, expanding window", fontsize=9)
    ax.grid(alpha=0.3)
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "oos_h1.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(n, n, figsize=(11, 9), sharex=True)
hz = np.arange(H_IRF + 1)
for i in range(n):
    for j in range(n):
        ax = axs[i, j]
        ax.plot(hz, G[:, i, j], color="#1f4e79")
        ax.fill_between(hz, G[:, i, j] - 2 * G_se[:, i, j], G[:, i, j] + 2 * G_se[:, i, j],
                        color="#1f4e79", alpha=0.2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{TARGETS[i]} to {TARGETS[j]}", fontsize=8)
fig.suptitle(f"Generalized IRFs, VARX({pf}) with oil, +/-2 bootstrap SE ({B_REP} reps)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "irf_generalized.png", dpi=150); plt.close(fig)

spot["full_sample_lag_criteria"] = crit.round(4).to_dict()
spot["full_sample_sc_lag"] = pf
spot["full_sample_varx_coefs"] = res.params.round(4).to_dict()
spot["unit_roots_adf_t"] = dict(zip(urt.series, urt.adf_t.round(4)))
spot["johansen_case3_trace"] = jo[1]["trace"]

summary = {
    "data_vintage": "fred_data/ as found at run time (reference run used the 2026-07-10 vintage)",
    "panel": {"start": qlab(dates[0]), "end": qlab(dates[-1]), "n": N, "first_origin": qlab(dates[I0]),
              "origins_h1": len(fc[1]), "origins_h4": len(fc[4])},
    "lag_choices_count": {f"h{h}": {str(k): v for k, v in lagrec[h].items()} for h in lagrec},
    "full_sample": {"sc_lag": pf, "max_inverse_root": float(inv_roots[0]),
                    "portmanteau_Q8_adj": [float(wt.test_statistic), float(wt.pvalue)],
                    "normality_JB": [float(nt.test_statistic), float(nt.pvalue)],
                    "white_joint": [float(white_lm), white_df, float(stats.chi2.sf(white_lm, white_df))],
                    "lm_serial": lm_rows, "white_components": white_comp},
    "johansen": jo, "glp_bvar": glp,
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "statsmodels": statsmodels.__version__,
                 "scipy": scipy.__version__},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
(OUT / "spotcheck_values.json").write_text(json.dumps(spot, indent=2, default=float))

pd.set_option("display.width", 200)
print("\nrelative RMSE vs RW (this run vs paper)")
print(rel[["h", "model"] + TARGETS + ["mean_vs_rw", "paper_mean", "mean_vs_ar", "origins"]].round(4).to_string(index=False))
print("\nClark-West vs RW, one-sided p")
print(cwt.pivot_table(index=["h", "model"], columns="target", values="p_one_sided", sort=False)[TARGETS].round(4).to_string())
print("\nBVAR minus OLS VARX, Diebold-Mariano (negative = BVAR better)")
print(dmt.round(4).to_string(index=False))
print("\nunit roots"); print(urt.round(3).to_string(index=False))
print("\nJohansen"); [print(" ", j["case"], j["trace"], j["crit5"], "rank", j["rank_5pct"]) for j in jo]
print(f"\nfull-sample VARX: SC lag {pf}, max inverse root {inv_roots[0]:.4f}, "
      f"Q(8) adj p {wt.pvalue:.4f}, JB p {nt.pvalue:.4f}, White joint p {stats.chi2.sf(white_lm, white_df):.4f}")
print("LM serial:", [(r["lag"], round(r["p_lag_h"], 4)) for r in lm_rows])
print("GLP:", json.dumps(glp))
print("lag choices:", summary["lag_choices_count"])
