"""Claim checks for macro-var.

Every result number stated in the README files and in the three paper sources (papers/macro_var/main.tex,
papers/var_selection_horserace/main.tex, studies/varx_permutation_study/paper/paper.tex)
is tied to a committed results file and compared with a tolerance of half a unit in the last printed
digit. Settings (dates, lag rule, max lags, sample bounds) are checked against the code. A coverage test
tokenises every number in those documents and fails on any number that is neither claimed nor
allowlisted (with a one-word reason). A claim whose text cannot be found, or whose source cannot be
read, fails; nothing is dropped.

Reads only committed files. Run:  pytest tests/test_claims.py
Print the claim table:           python tests/test_claims.py
"""
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
VARX = "studies/varx_permutation_study"
VR = f"{VARX}/results"
sys.path.insert(0, str(ROOT / VARX / "src"))

DOCS = {
    "README": ("README.md", "md"),
    "VARX-README": (f"{VARX}/README.md", "md"),
    "results-README": ("results/README.md", "md"),
    "papers-README": ("papers/README.md", "md"),
    "data-README": (f"{VARX}/data/README.md", "md"),
    "paper": ("papers/macro_var/main.tex", "tex"),
    "study1-paper": ("papers/var_selection_horserace/main.tex", "tex"),
    "varx-paper": (f"{VARX}/paper/paper.tex", "tex"),
}

# ----------------------------------------------------------------------------- tokeniser
NUM = re.compile(r"(?<![\w.])\d+(?:(?:\{,\}|,)\d{3})*(?:\.\d+)?")


def _blank(text, pattern, flags=0):
    return re.sub(pattern, lambda m: " " * len(m.group(0)), text, flags=flags)


def mask(text, kind):
    """Same-length copy of text with code, LaTeX commands and arguments, citations and URLs blanked."""
    if kind == "md":
        t = _blank(text, r"```.*?```", re.S)
        t = _blank(t, r"`[^`\n]*`")
        t = _blank(t, r"\]\([^)]*\)")
        return _blank(t, r"https?://\S+")
    t = _blank(text, r"\\begin\{thebibliography\}.*", re.S)
    t = _blank(t, r"(?m)(?<!\\)%.*$")
    t = _blank(t, r"\\(?:cite|ref|label|usepackage|documentclass|includegraphics|graphicspath)\*?(?:\[[^\]]*\])?(?:\{[^{}]*\}|\{\{[^{}]*\}\})+")
    t = _blank(t, r"\\begin\{(?:tabular|table|figure|figure\*|equation|abstract|IEEEkeywords)\*?\}(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})?")
    t = _blank(t, r"\\cmidrule\([a-z]+\)\{[^}]*\}")
    t = _blank(t, r"\\multicolumn\{\d+\}")
    t = _blank(t, r"\\(?:setlength|renewcommand)[^\n]*")
    t = _blank(t, r"\\(?:section|subsection)\{[^}]*\}")
    t = _blank(t, r"width=[\d.]*\\\w+")
    return t


RAW, FLAT, TOK = {}, {}, {}
for _n, (_p, _k) in DOCS.items():
    RAW[_n] = (ROOT / _p).read_text() if (ROOT / _p).exists() else ""
    FLAT[_n] = RAW[_n].replace("\n", " ")  # same length as RAW, so positions agree
    TOK[_n] = [(m.start(), m.end(), m.group(0)) for m in NUM.finditer(mask(RAW[_n], _k).replace("\n", " "))]

# ----------------------------------------------------------------------------- sources
_cache = {}


def J(path):
    if path not in _cache:
        _cache[path] = json.loads((ROOT / path).read_text())
    return _cache[path]


def P(path):
    if path not in _cache:
        _cache[path] = pd.read_parquet(ROOT / path)
    return _cache[path]


def CSV(path):
    if path not in _cache:
        _cache[path] = pd.read_csv(ROOT / path)
    return _cache[path]


def code(path):
    return (ROOT / path).read_text()


def codenum(path, pattern):
    m = re.search(pattern, code(path))
    if not m:
        raise AssertionError(f"pattern {pattern!r} not found in {path}")
    return float(m.group(1))


RR = "results/reproduce_results.json"
PS = f"{VR}/paper_stats.json"
RT = f"{VR}/rt_stats.json"
ID = f"{VR}/id_stats.json"
MP = f"{VR}/mp_stats.json"
CB = f"{VR}/combine_stats.json"
DN = f"{VR}/density_stats.json"
BY = f"{VR}/bayes_density_stats.json"


def A(spec, key):
    return lambda: {r["spec"]: r for r in J(RR)["phaseA_table"]}[spec][key]


def C(model, key):
    return lambda: {r["model"]: r for r in J(RR)["phaseC_table"]}[model][key]


def ps(path):
    def g():
        d = J(PS)
        for k in path.split("."):
            d = d[k]
        return d
    return g


# ----------------------------------------------------------------------------- claim registry
CLAIMS = []


def jp(path, dotted):
    def g():
        d = J(path)
        for k in dotted.split("."):
            d = d[k]
        return d
    return g


def A(spec, key):
    return lambda: {r["spec"]: r for r in J(RR)["phaseA_table"]}[spec][key]


def C(model, key):
    return lambda: {r["model"]: r for r in J(RR)["phaseC_table"]}[model][key]


def _span(doc, marker, after, until):
    base = FLAT[doc].find(after) if after else 0
    m = re.search(marker, FLAT[doc][max(base, 0):])
    if not m:
        return None
    lo = max(base, 0) + m.end()
    em = re.search(until, FLAT[doc][lo:])
    hi = lo + (em.end() if em else 400)
    return lo, hi


def _claim(doc, where, tok, src, field, getter, kind="eq", tol=None):
    CLAIMS.append({"doc": doc, "where": where, "pos": tok, "src": src, "field": field, "get": getter,
                   "kind": kind, "tol": tol})


def S(doc, marker, cells, until=r"\.(?:\s|\}|$)", after=None, kind="eq", tol=None):
    """Prose claims. cells = [(shown, src, field, getter), ...]; each `shown` is searched left to right after
    `marker` (a regex), up to the end of the sentence."""
    sp = _span(doc, marker, after, until)
    cur = sp[0] if sp else 0
    for shown, src, field, getter in cells:
        tok = None
        if sp:
            c = [t for t in TOK[doc] if t[0] >= cur and t[1] <= sp[1] and t[2] == shown]
            tok = c[0] if c else None
        _claim(doc, marker, tok, src, field, getter, kind, tol)
        if tok:
            cur = tok[1]


def T(doc, marker, getters, after=None):
    """Table row: every numeric token after `marker` up to the row end `\\\\` is paired, in order, with one
    (src, field, getter); None leaves that token to the allowlist."""
    sp = _span(doc, marker, after, r"\\\\")
    toks = [t for t in TOK[doc] if sp and sp[0] <= t[0] and t[1] <= sp[1]]
    n = max(len(toks), len(getters))
    for i in range(n):
        tok = toks[i] if i < len(toks) else None
        g = getters[i] if i < len(getters) else None
        if g is None:
            continue
        _claim(doc, marker, tok if i < len(toks) else None, *g)


def g(src, field, getter, kind="eq", tol=None):
    return (src, field, getter, kind, tol)


def cell(shown, src, field, getter):
    return (shown, src, field, getter)


RR = "results/reproduce_results.json"
PS = f"{VR}/paper_stats.json"
RT = f"{VR}/rt_stats.json"
ID = f"{VR}/id_stats.json"
MP = f"{VR}/mp_stats.json"
CB = f"{VR}/combine_stats.json"
DN = f"{VR}/density_stats.json"
BY = f"{VR}/bayes_density_stats.json"
RTC = f"{VR}/rt_compare.parquet"
CBC = f"{VR}/combine_compare.parquet"
XC = "results/crosscheck"
VL = f"{VARX}/src/varx_lab.py"
Q = "CORE+OIL(x)"


def varx_lab():
    import importlib
    return importlib.import_module("varx_lab")


def n_endog():
    return len({tuple(c["endog"]) for c in varx_lab().enumerate_specs()})


def n_shocks():
    return len({tuple(c["shocks"]) for c in varx_lab().enumerate_specs()})


def adv12():
    b = J(PS)["gdp_1_all"]
    return 100 * (1 - b["best"]["rmse"] / b["noshock_best_rmse"])


def share():
    return 100 * J(PS)["dm_sig_share_5pct"]


def share_below_core():
    return 100 * J(PS)["gdp1_vs_core"]["share_below_core"]


def share_better():
    return 100 * J(PS)["dm_better_share_5pct"]


def rtg(k, s="all", h="1"):
    return J(RT)["gdp"][h][s][k]


def degr(spec):
    d = P(RTC)
    return float(d[(d.spec == spec) & (d.target == "gdp") & (d.h == 1) & (d["sample"] == "all")].iloc[0].degr_pct)


def rtcol(spec, col):
    d = P(RTC)
    return float(d[(d.spec == spec) & (d.target == "gdp") & (d.h == 1) & (d["sample"] == "all")].iloc[0][col])


def idv(t, s, a, h="1"):
    return J(ID)[t][h][s][a]["rmse"]


def mpv(t, h, s, a):
    return J(MP)[t][h][s][a]["rmse"]


CELLS = [(t, h, s) for t in ("gdp", "inf") for h in ("1", "4", "8") for s in ("all", "excovid")]


def blend():
    j = J(CB)
    return [j[m][t][k] for m in j for t in j[m] for k in ("equal_vs_best_mean_pct", "invmse_vs_best_mean_pct")]


def gap(h, tr):
    d = J(PS)["path_median_gdp"][f"h{h}_{tr}"]
    return d["uncond"] - d["realized"]


def paper_claims(pp, study1, varx):
    """The claims for one paper source: Study 1 sections if study1, VARX sections if varx."""
    if study1:
        # ============================================================================ Study 1 (paper)
        S(pp, r"The common sample is", [cell("128", RR, "sample_level.n", jp(RR, "sample_level.n")),
                                        cell("125", RR, "sample_yoy.n", jp(RR, "sample_yoy.n"))], until=r"span 125 quarters")
        S(pp, r"whiteness; parsimony, a ratio", [])
        S(pp, r"a portmanteau test with", [cell("0.01", "src/var_horserace.py", "whiteness threshold",
                                               lambda: codenum("src/var_horserace.py", r'whiteness_p"\]\s*>\s*([\d.]+)'))])
        S(pp, r"attains a mean relative RMSE of", [cell("0.97", RR, "mean_h1", A(Q, "mean_h1")), cell("0.88", RR, "mean_h4", A(Q, "mean_h4"))])
        S(pp, r"Per target at", [cell("0.84", RR, "GDP_h1", A(Q, "GDP_h1")), cell("1.07", RR, "INF_h1", A(Q, "INF_h1")),
                                 cell("1.13", RR, "POLICY_h1", A(Q, "POLICY_h1")), cell("0.83", RR, "FX_h1", A(Q, "FX_h1"))])
        for shown, k, kind in (("0.001", "FX", "lt"), ("0.013", "POLICY", "eq"), ("0.081", "GDP", "eq"), ("0.176", "INF", "eq")):
            S(pp, r"Clark-West tests against the random walk", [cell(shown, RR, f"phaseA_CW_COREOIL_vs_RW.{k}.p",
                                                                      jp(RR, f"phaseA_CW_COREOIL_vs_RW.{k}.p"))], kind=kind, until=r"\.\s")
        S(pp, r"which lowers the mean relative RMSE to", [cell("0.95", RR, "CORE+WEALTH.mean_h1", A("CORE+WEALTH", "mean_h1")),
                                                         cell("0.91", RR, "CORE+WEALTH+OIL(x).mean_h1", A("CORE+WEALTH+OIL(x)", "mean_h1"))])
        S(pp, r"fails the parsimony filter, with an", [cell("2.3", RR, "CORE+WEALTH.T_over_k", A("CORE+WEALTH", "T_over_k")),
                                                      cell("2.1", RR, "CORE+WEALTH+OIL(x).T_over_k", A("CORE+WEALTH+OIL(x)", "T_over_k")),
                                                      cell("0.04", RR, "CORE+WEALTH+OIL(x).whiteness_p", A("CORE+WEALTH+OIL(x)", "whiteness_p"))])
        S(pp, r"Utilization-adjusted TFP as an exogenous", [cell("1.06", RR, "CORE+TFP(x).mean_h1", A("CORE+TFP(x)", "mean_h1"))])
        for lab, spec in ((r"CORE\s+&", "CORE"), (r"CORE\+OIL\(x\) \\textit\{\(sel\.\)\} &", Q), (r"CORE\+TFP\(x\)\s+&", "CORE+TFP(x)"),
                          (r"CORE\+WEALTH\s+&", "CORE+WEALTH"), (r"CORE\+WEALTH\+OIL\(x\)\s+&", "CORE+WEALTH+OIL(x)")):
            gs = [g(RR, f"{spec}.mean_h1", A(spec, "mean_h1")), g(RR, f"{spec}.T_over_k", A(spec, "T_over_k"))]
            if spec == "CORE+WEALTH+OIL(x)":
                gs.append(g(RR, f"{spec}.whiteness_p", A(spec, "whiteness_p")))
            T(pp, lab, gs, after="Baseline specification search, $h=1$")
        S(pp, r"The two series correlate at only", [cell("0.52", RR, "hp.corr", jp(RR, "hp.corr")),
                                                   cell("0.86", RR, "hp.mean_abs_rev", jp(RR, "hp.mean_abs_rev")),
                                                   cell("0.71", RR, "hp.last8_abs_rev", jp(RR, "hp.last8_abs_rev"))], until=r"sixty-six")
        S(pp, r"The two cores are close, with mean relative", [cell("0.915", RR, "level_mean3_h1", jp(RR, "gap_vs_level.level_mean3_h1")),
                                                              cell("0.927", RR, "gap_mean3_h1", jp(RR, "gap_vs_level.gap_mean3_h1"))])
        S(pp, r"The gap improves only the rate's own", [cell("0.75", RR, "gap_rate_own", jp(RR, "gap_vs_level.gap_rate_own")),
                                                       cell("1.13", RR, "level_rate_own", jp(RR, "gap_vs_level.level_rate_own"))])
        S(pp, r"The univariate AR is the single best", [cell("0.85", RR, "AR(BIC).mean_h1", C("AR(BIC)", "mean_h1"))])
        S(pp, r"The small-system estimators cluster", [
            cell("0.88", RR, "min small mean_h1", lambda: min(r["mean_h1"] for r in J(RR)["phaseC_table"] if r["model"].endswith(":small"))),
            cell("0.90", RR, "max small mean_h1", lambda: max(r["mean_h1"] for r in J(RR)["phaseC_table"] if r["model"].endswith(":small")))])
        S(pp, r"the best single output forecast, at", [cell("0.69", RR, "BVAR-Tight:small.GDP_h1", C("BVAR-Tight:small", "GDP_h1"))])
        for lab, model in ((r"AR\(BIC\)\s+& --\s+&", "AR(BIC)"), (r"Ridge\s+& small\s+&", "Ridge:small"),
                           (r"BVAR-Loose\s+& small\s+&", "BVAR-Loose:small"), (r"OLS\s+& small\s+&", "OLS:small"),
                           (r"BVAR-Tight\s+& small\s+&", "BVAR-Tight:small"), (r"Elastic-net\s+& small\s+&", "ENet:small"),
                           (r"LASSO\s+& small\s+&", "LASSO:small"), (r"BVAR-Tight\s+& wide\s+&", "BVAR-Tight:wide"),
                           (r"Ridge\s+& wide\s+&", "Ridge:wide"), (r"Random walk\s+& --\s+&", "RW"),
                           (r"LASSO\s+& wide\s+&", "LASSO:wide"), (r"Elastic-net\s+& wide\s+&", "ENet:wide"),
                           (r"OLS \(overfit\)\s+& wide\s+&", "OLS:wide(overfit)")):
            T(pp, lab, [g(RR, f"{model}.mean_h1", C(model, "mean_h1")), g(RR, f"{model}.mean_h4", C(model, "mean_h4"))],
              after="Model-class comparison, mean relRMSE")
        S(pp, r"augmented Dickey-Fuller", [cell("0.81", RR, "phaseD_adf.INF_yoy", jp(RR, "phaseD_adf.INF_yoy"))])
        for lab, t in ((r"Output\s+&", "GDP"), (r"Inflation\s+&", "INF"), (r"Policy \(level\)\s+&", "POLICY"), (r"FX\s+&", "FX")):
            T(pp, lab, [g(RR, f"phaseD_table4_CORE.{t}.{k}", jp(RR, f"phaseD_table4_CORE.{t}.{k}"))
                        for k in ("aq_h1", "aq_h4", "yoy_h1", "yoy_h4")], after="Transformation robustness, relRMSE")

    if varx:
        if pp == "varx-paper":  # the stand-alone VARX abstract words these sentences differently
            S(pp, r"The design crosses", [cell("16", VL, "endogenous sets", n_endog), cell("12", VL, "shock subsets", n_shocks),
                                          cell("1{,}632", PS, "counts.spec_rows", jp(PS, "counts.spec_rows")),
                                          cell("1{,}014", PS, "counts.unique_models", jp(PS, "counts.unique_models"))])
            S(pp, r"the best GDP specification cuts", [cell("12", PS, "1-best/noshock gdp_1_all", adv12)])
            S(pp, r"Fourth, almost nothing is statistically distinguishable:", [cell("66", PS, "gdp1_vs_core.share_below_core", share_below_core)], until=r"percent level")
            S(pp, r"survives publication lags and national-accounts revisions almost intact, from", [
                cell("16.6", RT, "gdp.1.all.adv_final_pct", lambda: rtg("adv_final_pct")),
                cell("15.8", RT, "gdp.1.all.adv_rt_pct", lambda: rtg("adv_rt_pct"))])
        else:
            S(pp, r"which crosses", [cell("16", VL, "endogenous sets", n_endog), cell("12", VL, "shock subsets", n_shocks),
                                     cell("1,632", PS, "counts.spec_rows", jp(PS, "counts.spec_rows")),
                                     cell("1,014", PS, "counts.unique_models", jp(PS, "counts.unique_models"))])
            S(pp, r"In the VARX study, exogenous shocks improve", [cell("12", PS, "1-best/noshock gdp_1_all", adv12)])
            S(pp, r"forecasting tool\.", [cell("66", PS, "gdp1_vs_core.share_below_core", share_below_core)], until=r"percent level")
            S(pp, r"A real-time replication on ALFRED vintages leaves", [cell("16.6", RT, "gdp.1.all.adv_final_pct", lambda: rtg("adv_final_pct")),
                                                                      cell("15.8", RT, "gdp.1.all.adv_rt_pct", lambda: rtg("adv_rt_pct"))])
        S(pp, r"All series are quarterly, 1990Q1 to 2026Q1", [cell("145", PS, "counts.n_obs", jp(PS, "counts.n_obs"))], until=r"\)")
        S(pp, r"minimum Ljung-Box whiteness p-value above", [cell("0.01", VL, "validity lb_min", lambda: codenum(VL, r"lb_min=([\d.]+)"))])
        S(pp, r"The gates keep", [cell("85.6", PS, "counts.valid_share", lambda: 100 * J(PS)["counts"]["valid_share"])])
        S(pp, r"Forecasts are iterated from", [cell("65", PS, "counts.n_origins", jp(PS, "counts.n_origins"))])
        for lab, key in (("GDP, 1", "gdp_1_all"), ("GDP, 4", "gdp_4_all"), ("GDP, 8", "gdp_8_all"), ("Inf, 1", "inf_1_all"),
                         ("Inf, 4", "inf_4_all"), ("Stance, 1", "stance_1_all"), ("Stance, 4", "stance_4_all"),
                         ("Stance, 8", "stance_8_all"), ("FX, 1", "fx_1_all")):
            T(pp, re.escape(lab) + r" &", [g(PS, f"{key}.best.p", jp(PS, f"{key}.best.p")), g(PS, f"{key}.best.rmse", jp(PS, f"{key}.best.rmse")),
                                           g(PS, f"{key}.best.r2_oos", jp(PS, f"{key}.best.r2_oos")),
                                           g(PS, f"{key}.noshock_best_rmse", jp(PS, f"{key}.noshock_best_rmse"))],
              after="Best specification per target and horizon")
        S(pp, r"GDP at \$h=1\$ is genuinely forecastable", [cell("0.33", PS, "gdp_1_all.best.r2_oos", jp(PS, "gdp_1_all.best.r2_oos")),
                                                            cell("0.10", PS, "gdp_1_excovid.best.r2_oos", jp(PS, "gdp_1_excovid.best.r2_oos"))])
        S(pp, r"Inflation and the dollar are barely forecastable", [cell("0.08", PS, "inf_1_all.best.r2_oos", jp(PS, "inf_1_all.best.r2_oos")),
                                                                   cell("0.05", PS, "fx_1_all.best.r2_oos", jp(PS, "fx_1_all.best.r2_oos"))])
        S(pp, r"The stance loses to its own AR", [cell("0.10", PS, "stance_8_all.best.r2_oos", jp(PS, "stance_8_all.best.r2_oos"))])
        S(pp, r"AIC, which picks longer lags", [cell("1.62", PS, "best_by_rule.stance_8_aic", jp(PS, "best_by_rule.stance_8_aic")),
                                                cell("2.01", PS, "best_by_rule.stance_8_bic", jp(PS, "best_by_rule.stance_8_bic"))])
        S(pp, r"falls monotonically in the number of included shocks", [
            cell("6.76", PS, "shockcount_gdp_h1.0.median", jp(PS, "shockcount_gdp_h1.0.median")),
            cell("6.61", PS, "shockcount_gdp_h1.1.median", jp(PS, "shockcount_gdp_h1.1.median")),
            cell("6.30", PS, "shockcount_gdp_h1.2.median", jp(PS, "shockcount_gdp_h1.2.median")),
            cell("5.81", PS, "shockcount_gdp_h1.4.median", jp(PS, "shockcount_gdp_h1.4.median"))], until=r"\(Fig")
        tm = lambda t, k: jp(PS, f"treat_median_h1.{t}.{k}")
        S(pp, r"Raw shocks win for GDP", [cell("6.05", PS, "treat gdp raw", tm("gdp", "raw")), cell("6.45", PS, "treat gdp gdp_resid", tm("gdp", "gdp_resid")),
                                          cell("6.51", PS, "treat gdp full_resid", tm("gdp", "full_resid"))], until=r"\)")
        for lab, t in ((r"GDP\s*&", "gdp"), (r"Inflation\s*&", "inf"), (r"Stance\s*&", "stance"), (r"FX\s*&", "fx")):
            T(pp, lab, [g(PS, f"treat_median_h1.{t}.{k}", tm(t, k)) for k in ("raw", "gdp_resid", "full_resid")],
              after="by shock treatment, BIC lag rule")
        S(pp, r"BIC selects \$p \\in", [cell("227", PS, "p_by_rule.bic.1", jp(PS, "p_by_rule.bic.1")), cell("317", PS, "p_by_rule.bic.2", jp(PS, "p_by_rule.bic.2"))],
          until=r"specifications\)")
        lm = lambda k, r: jp(PS, f"lagrule_median.{k}.{r}")
        S(pp, r"On median RMSE at \$h=1\$ over all three rules", [
            cell("6.44", PS, "gdp_1.bic", lm("gdp_1", "bic")), cell("6.62", PS, "gdp_1.aic", lm("gdp_1", "aic")),
            cell("0.96", PS, "inf_1.bic", lm("inf_1", "bic")), cell("1.04", PS, "inf_1.aic", lm("inf_1", "aic")),
            cell("9.67", PS, "fx_1.bic", lm("fx_1", "bic")), cell("10.32", PS, "fx_1.aic", lm("fx_1", "aic")),
            cell("0.550", PS, "stance_1.aic", lm("stance_1", "aic")), cell("0.554", PS, "stance_1.bic", lm("stance_1", "bic"))], until=r"\),\swhere")
        S(pp, r"The headline results above do not depend", [cell("5.641", PS, "best_by_rule.gdp_1_bic", jp(PS, "best_by_rule.gdp_1_bic")),
                                                          cell("5.641", PS, "best_by_rule.gdp_1_hqic", jp(PS, "best_by_rule.gdp_1_hqic")),
                                                          cell("5.660", PS, "best_by_rule.gdp_1_aic", jp(PS, "best_by_rule.gdp_1_aic"))])
        S(pp, r"conditioning on the realized path adds little", [cell("0.18", PS, "path_median_gdp h1_raw gap", lambda: gap(1, "raw"))])
        S(pp, r"By \$h=8\$ the realized path is worth between", [
            cell("0.21", PS, "min h8 gap", lambda: min(gap(8, t) for t in ("raw", "gdp_resid", "full_resid"))),
            cell("0.34", PS, "max h8 gap", lambda: max(gap(8, t) for t in ("raw", "gdp_resid", "full_resid")))])
        S(pp, r"modest relative to the roughly", [cell("6", PS, "path_median_gdp.h1_raw.uncond", jp(PS, "path_median_gdp.h1_raw.uncond"))], tol=0.5)
        for i, (h, tr) in enumerate([(h, tr) for h in (1, 4, 8) for tr in ("raw", "gdp_resid", "full_resid")]):
            lab = {"raw": "raw", "gdp_resid": "GDP-purged", "full_resid": "fully purged"}[tr]
            T(pp, rf"\b{h} & {re.escape(lab)} &", [g(PS, f"path_median_gdp.h{h}_{tr}.uncond", jp(PS, f"path_median_gdp.h{h}_{tr}.uncond")),
                                                g(PS, f"path_median_gdp.h{h}_{tr}.realized", jp(PS, f"path_median_gdp.h{h}_{tr}.realized")),
                                                g(PS, f"gap h{h} {tr}", (lambda h=h, tr=tr: gap(h, tr)))],
              after="Median GDP RMSE by exogenous path assumption")
        S(pp, r"Across all validity-gated specifications, targets, and horizons under BIC", [cell("3.8", PS, "dm_sig_share_5pct", share)])
        if True:  # both papers restate the DM result one-sided
            S(pp, r"the worse one: only", [cell("2", PS, "dm_n_better_5pct", jp(PS, "dm_n_better_5pct")),
                                          cell("2,826", PS, "dm_tests", jp(PS, "dm_tests")),
                                          cell("0.07", PS, "dm_better_share_5pct", share_better)], until=r"one-sided\.")
            S(pp, r"For GDP at h=1,", [cell("66", PS, "gdp1_vs_core.share_below_core", share_below_core),
                                      cell("314", PS, "gdp1_vs_core.n_specs", jp(PS, "gdp1_vs_core.n_specs"))], until=r"significantly\.")
        S(pp, r"rejection p-value is", [cell("0.38", PS, "gdp_1_all.best.dm_p", jp(PS, "gdp_1_all.best.dm_p")),
                                        cell("65", PS, "counts.n_origins", jp(PS, "counts.n_origins"))], until=r"evaluation points")
        S(pp, r"Do not report full-window GDP gains", [cell("0.33", PS, "gdp_1_all.best.r2_oos", jp(PS, "gdp_1_all.best.r2_oos")),
                                                      cell("0.10", PS, "gdp_1_excovid.best.r2_oos", jp(PS, "gdp_1_excovid.best.r2_oos"))])
        S(pp, r"data vintage current roughly", [cell("45", "src varx_realtime.py", "VLAG_DAYS", lambda: codenum(f"{VARX}/src/varx_realtime.py", r"VLAG_DAYS = (\d+)"))])
        S(pp, r"relative advantage of the all-four-shock system", [
            cell("16.6", RT, "adv_final_pct", lambda: rtg("adv_final_pct")), cell("15.8", RT, "adv_rt_pct", lambda: rtg("adv_rt_pct")),
            cell("7.5", RT, "excovid adv_final_pct", lambda: rtg("adv_final_pct", "excovid")), cell("7.8", RT, "excovid adv_rt_pct", lambda: rtg("adv_rt_pct", "excovid")),
            cell("0.42", RT, "all4_r2_rt", lambda: rtg("all4_r2_rt"))])
        S(pp, r"so the GDP-purged system degrades by", [cell("35", RTC, "all4_gdp_resid degr_pct", lambda: degr("all4_gdp_resid")),
                                                       cell("6", RTC, "all4_raw degr_pct", lambda: degr("all4_raw"))])
        for lab, a_, b_ in (("Pure VAR", "var_final", "var_rt"), ("All four, raw", "all4_final", "all4_rt")):
            T(pp, rf"\b1 & {lab} &", [g(RT, f"gdp.1.all.{a_}", lambda a_=a_: rtg(a_)), g(RT, f"gdp.1.all.{b_}", lambda b_=b_: rtg(b_)),
                                     g(RT, f"gdp.1.excovid.{a_}", lambda a_=a_: rtg(a_, "excovid")), g(RT, f"gdp.1.excovid.{b_}", lambda b_=b_: rtg(b_, "excovid"))],
              after="GDP forecast RMSE, final vintage against real-time")
        T(pp, r"\b1 & \\emph\{Advantage\} &", [g(RT, "adv_final_pct all", lambda: rtg("adv_final_pct")), g(RT, "adv_rt_pct all", lambda: rtg("adv_rt_pct")),
                                              g(RT, "adv_final_pct excovid", lambda: rtg("adv_final_pct", "excovid")), g(RT, "adv_rt_pct excovid", lambda: rtg("adv_rt_pct", "excovid"))],
          after="GDP forecast RMSE, final vintage against real-time")
        T(pp, r"\b1 & All four, GDP-purged &", [g(RTC, "all4_gdp_resid rmse_final", lambda: rtcol("all4_gdp_resid", "rmse_final")),
                                               g(RTC, "all4_gdp_resid rmse_rt", lambda: rtcol("all4_gdp_resid", "rmse_rt"))],
          after="GDP forecast RMSE, final vintage against real-time")
        S(pp, r"On the full window at \$h=1\$ the raw proxy cuts inflation RMSE by", [
            cell("7.6", ID, "inf h1 all oil_raw gain %", lambda: 100 * (1 - idv("inf", "all", "oil_raw") / idv("inf", "all", "pure_var"))),
            cell("5.0", ID, "oil_news gain %", lambda: 100 * (1 - idv("inf", "all", "oil_news") / idv("inf", "all", "pure_var"))),
            cell("2.8", ID, "oil_surp gain %", lambda: 100 * (1 - idv("inf", "all", "oil_surp") / idv("inf", "all", "pure_var")))])
        S(pp, r"the raw proxy beats the pure VAR in", [
            cell("8", ID, "cells oil_raw<pure_var", lambda: sum(idv(t, s, "oil_raw", h) < idv(t, s, "pure_var", h) for t, h, s in CELLS)),
            cell("4", ID, "cells oil_news<pure_var", lambda: sum(idv(t, s, "oil_news", h) < idv(t, s, "pure_var", h) for t, h, s in CELLS))])
        for lab, arm in (("Pure VAR \\(none\\)", "pure_var"), ("Raw growth proxy", "oil_raw"), ("HF surprise instrument", "oil_surp"),
                         ("Identified news shock", "oil_news")):
            T(pp, lab + r" &", [g(ID, f"{t}.1.{s}.{arm}.rmse", (lambda t=t, s=s, arm=arm: idv(t, s, arm))) for t in ("gdp", "inf") for s in ("all", "excovid")],
              after="with one exogenous oil regressor")
        S(pp, r"Table~\\ref\{tab:mp\} shows the raw surprise helps", [
            cell("1", MP, "inf 4 all 1-mp_surp/pure_var %", lambda: 100 * (1 - mpv("inf", "4", "all", "mp_surp") / mpv("inf", "4", "all", "pure_var")))], tol=0.5)
        S(pp, r"the orthogonalized surprise, the cleaner shock, is worse than the raw one in", [
            cell("11", MP, "cells mp_orth>mp_surp", lambda: sum(mpv(t, h, s, "mp_orth") > mpv(t, h, s, "mp_surp") for t, h, s in CELLS))])
        for lab, arm in (("Core only \\(stance\\)", "pure_var"), ("Fed funds change", "mp_raw"), ("Raw MPS surprise", "mp_surp"),
                         ("Orthogonalized MPS", "mp_orth")):
            T(pp, lab + r" &", [g(MP, f"{t}.{h}.{s}.{arm}.rmse", (lambda t=t, h=h, s=s, arm=arm: mpv(t, h, s, arm)))
                               for (t, h) in (("gdp", "1"), ("inf", "4")) for s in ("all", "excovid")],
              after="Monetary margin: forecast RMSE")
        S(pp, r"Combining the proxy and identified forecasts", [
            cell("0.2", CB, "min blend pct", lambda: min(blend())), cell("0.8", CB, "max blend pct", lambda: max(blend()))], until=r"percent worse")
        S(pp, r"encompasses the identified one in", [
            cell("21", CBC, "beta_p>=0.05 cells", lambda: int((P(CBC).beta_p >= 0.05).sum())), cell("24", CBC, "cells", lambda: len(P(CBC)))], until=r"cells")
        nd = lambda arm, t, k: jp(DN, f"{arm}.{t}.1.all.{'pure_var' if False else k}")
        S(pp, r"covers only", [cell("68", DN, "naive gdp 1 all all4_raw cov90", lambda: 100 * J(DN)["naive"]["gdp"]["1"]["all"]["all4_raw"]["cov90"]),
                               cell("88", DN, "naive gdp 1 all pure_var cov90", lambda: 100 * J(DN)["naive"]["gdp"]["1"]["all"]["pure_var"]["cov90"]),
                               cell("13", DN, "naive gdp 1 all all4_raw log_score", lambda: -J(DN)["naive"]["gdp"]["1"]["all"]["all4_raw"]["log_score"])],
          until=r"minus \d+")
        S(pp, r"restores GDP coverage to", [cell("91", DN, "fair gdp 1 all all4_raw cov90", lambda: 100 * J(DN)["fair"]["gdp"]["1"]["all"]["all4_raw"]["cov90"])])
        for lab, t, arm in (("GDP & Pure VAR", "gdp", "pure_var"), ("GDP & All four", "gdp", "all4_raw"),
                            ("Inflation & Pure VAR", "inf", "pure_var"), ("Inflation & All four", "inf", "all4_raw")):
            T(pp, lab + r" &", [g(DN, f"naive.{t}.1.all.{arm}.cov90", (lambda t=t, arm=arm: J(DN)["naive"][t]["1"]["all"][arm]["cov90"])),
                               g(DN, f"fair.{t}.1.all.{arm}.cov90", (lambda t=t, arm=arm: J(DN)["fair"][t]["1"]["all"][arm]["cov90"])),
                               g(DN, f"naive.{t}.1.all.{arm}.log_score", (lambda t=t, arm=arm: J(DN)["naive"][t]["1"]["all"][arm]["log_score"])),
                               g(DN, f"fair.{t}.1.all.{arm}.log_score", (lambda t=t, arm=arm: J(DN)["fair"][t]["1"]["all"][arm]["log_score"]))],
              after="Predictive density at $h=1$, full window")
        bg = lambda t, k: (lambda: J(BY)[t]["all"]["1"][k]["cov90"])
        S(pp, r"lifts the pure VAR's", [cell("0.88", BY, "gdp all 1 plugin+gauss cov90", bg("gdp", "plugin+gauss")),
                                        cell("0.91", BY, "gdp all 1 bayes+gauss cov90", bg("gdp", "bayes+gauss")),
                                        cell("0.83", BY, "inf all 1 plugin+gauss cov90", bg("inf", "plugin+gauss")),
                                        cell("0.85", BY, "inf all 1 bayes+gauss cov90", bg("inf", "bayes+gauss"))], until=r"at one quarter")
        S(pp, r"it lowers coverage from", [cell("0.83", BY, "inf all 1 plugin+gauss cov90", bg("inf", "plugin+gauss")),
                                           cell("0.77", BY, "inf all 1 plugin+emp cov90", bg("inf", "plugin+emp"))], until=r"to 0\.\d+")



paper_claims("paper", True, True)
paper_claims("study1-paper", True, False)
paper_claims("varx-paper", False, True)


# ============================================================================ READMEs
R = "README"
S(R, r"Across more than", [cell("1,000", PS, "counts.unique_models > 1000", jp(PS, "counts.unique_models"))], kind="gt")
S(R, r"A permutation study that runs", [cell("1,014", PS, "counts.unique_models", jp(PS, "counts.unique_models"))])
S(R, r"Study 1 picks the VAR lag by BIC, up to", [cell("4", "src/var_horserace.py", "MAXLAG_VAR", lambda: codenum("src/var_horserace.py", r"MAXLAG_VAR = (\d+)"))])
S(R, r"and fixes the lag at", [cell("2", "src/reproduce.py", "P", lambda: codenum("src/reproduce.py", r"\nP = (\d+)"))])
S(R, r"initial training window \(to 2009Q4\), over", [cell("1", VL, "min P_GRID", lambda: min(varx_lab().P_GRID)),
                                                     cell("4", VL, "max P_GRID", lambda: max(varx_lab().P_GRID))], until=r"lags")
S(R, r"The greedy forward path", [cell("1.0431", RR, "CORE.mean_h1", A("CORE", "mean_h1")), cell("0.9452", RR, "CORE+WEALTH.mean_h1", A("CORE+WEALTH", "mean_h1")),
                                  cell("0.9117", RR, "CORE+WEALTH+OIL(x).mean_h1", A("CORE+WEALTH+OIL(x)", "mean_h1"))], until=r"at h=1\.")
S(R, r"Neither wealth\s+specification is admissible", [cell("2.29", RR, "CORE+WEALTH.T_over_k", A("CORE+WEALTH", "T_over_k")),
                                                       cell("2.10", RR, "CORE+WEALTH+OIL(x).T_over_k", A("CORE+WEALTH+OIL(x)", "T_over_k"))], until=r"The eligible")
S(R, r"The eligible winner, used throughout", [cell("0.9679", RR, "mean_h1", A(Q, "mean_h1")), cell("0.8786", RR, "mean_h4", A(Q, "mean_h4"))], until=r"The rejected")
S(R, r"The rejected", [cell("0.9117", RR, "CORE+WEALTH+OIL(x).mean_h1", A("CORE+WEALTH+OIL(x)", "mean_h1"))])
S(R, r"The heatmap is drawn from the", [
    cell("57", f"{XC}/relrmse.csv", "origins h1 VARX", lambda: CSV(f"{XC}/relrmse.csv").query("h==1 and model=='VARX_CORE+OIL'").origins.iloc[0]),
    cell("0.9657", "results/var_horserace_results.csv", "CORE+OIL(x) mean_relRMSE_h1",
         lambda: CSV("results/var_horserace_results.csv").set_index("model_id").loc[Q, "mean_relRMSE_h1"])], until=r"at h=1\.")
S(R, r"The text uses\s+the", [cell("54", RR, "oos.n_origins_h1", jp(RR, "oos.n_origins_h1")), cell("0.9679", RR, "mean_h1", A(Q, "mean_h1"))])
S(R, r"the two-sided cycle\s+differs", [cell("0.86", RR, "hp.mean_abs_rev", jp(RR, "hp.mean_abs_rev")),
                                       cell("80", RR, "hp mean_abs_rev/cyc_std %", lambda: 100 * J(RR)["hp"]["mean_abs_rev"] / J(RR)["hp"]["cyc_std"])], until=r"standard\s+deviation")
S(R, r"Over the last eight quarters the gap is", [cell("65.8", RR, "hp.pct_of_std", jp(RR, "hp.pct_of_std"))])
S(R, r"VARX permutation study\.", [cell("1,632", PS, "counts.spec_rows", jp(PS, "counts.spec_rows")), cell("544", PS, "counts.cells", jp(PS, "counts.cells")),
                                   cell("1,014", PS, "counts.unique_models", jp(PS, "counts.unique_models"))], until=r"from 2009Q4")
S(R, r"Exogenous shocks buy a", [cell("12", PS, "1-best/noshock gdp_1_all", adv12),
                                 cell("2", PS, "dm_n_better_5pct", jp(PS, "dm_n_better_5pct")),
                                 cell("2,826", PS, "dm_tests", jp(PS, "dm_tests"))], until=r"other way\.")
S(R, r"The gain survives ALFRED", [cell("16.6", RT, "adv_final_pct", lambda: rtg("adv_final_pct")), cell("15.8", RT, "adv_rt_pct", lambda: rtg("adv_rt_pct"))])
S(R, r"with the lag chosen by BIC \(up", [cell("4", "src/var_horserace.py", "MAXLAG_VAR", lambda: codenum("src/var_horserace.py", r"MAXLAG_VAR = (\d+)")),
                                         cell("0.9679", RR, "mean_h1", A(Q, "mean_h1"))])
S(R, r"The Minnesota BVAR in the cross-check averages", [
    cell("0.901", f"{XC}/relrmse.csv", "BVAR h1 mean_vs_rw", lambda: CSV(f"{XC}/relrmse.csv").query("h==1 and model=='BVAR_CORE+OIL'").mean_vs_rw.iloc[0]),
    cell("0.966", f"{XC}/relrmse.csv", "VARX h1 mean_vs_rw", lambda: CSV(f"{XC}/relrmse.csv").query("h==1 and model=='VARX_CORE+OIL'").mean_vs_rw.iloc[0]),
    cell("57", f"{XC}/relrmse.csv", "origins h1", lambda: CSV(f"{XC}/relrmse.csv").query("h==1 and model=='VARX_CORE+OIL'").origins.iloc[0])], until=r"origins,")
S(R, r"no target beats the OLS VARX in a\s+Diebold-Mariano test", [
    cell("0.18", f"{XC}/bvar_vs_ols_dm.csv", "min p", lambda: CSV(f"{XC}/bvar_vs_ols_dm.csv").p_two_sided.min()),
    cell("0.81", f"{XC}/bvar_vs_ols_dm.csv", "max p", lambda: CSV(f"{XC}/bvar_vs_ols_dm.csv").p_two_sided.max())], until=r"\)")
S(R, r"The full-sample VARX\(", [cell("2", f"{XC}/summary.json", "full_sample.sc_lag", jp(f"{XC}/summary.json", "full_sample.sc_lag")),
                                cell("0.89", f"{XC}/summary.json", "max_inverse_root", jp(f"{XC}/summary.json", "full_sample.max_inverse_root")),
                                cell("0.034", f"{XC}/summary.json", "lm_serial[0].p_lag_h", lambda: J(f"{XC}/summary.json")["full_sample"]["lm_serial"][0]["p_lag_h"])],
  until=r"LM p = [\d.]+")
S(R, r"The shock models'", [cell("12", PS, "1-best/noshock gdp_1_all", adv12)], until=r"COVID")
S(R, r"lower\s+h=8 RMSE than BIC does", [cell("1.62", PS, "best_by_rule.stance_8_aic", jp(PS, "best_by_rule.stance_8_aic")),
                                       cell("2.01", PS, "best_by_rule.stance_8_bic", jp(PS, "best_by_rule.stance_8_bic"))], until=r"\)")
prov = lambda k: (lambda: J(RR)["provenance"][k])
S(R, r"reference run \(numpy", [cell("2.4", RR, "provenance.numpy", prov("numpy")), cell("3.0", RR, "provenance.pandas", prov("pandas")),
                                cell("0.14", RR, "provenance.statsmodels", prov("statsmodels")), cell("1.8", RR, "provenance.sklearn", prov("sklearn")),
                                cell("8.0", RR, "provenance.arch", prov("arch"))], until=r"\);", kind="prefix")
S(R, r"on a fresh FRED pull \(", [cell("1,014", PS, "counts.unique_models", jp(PS, "counts.unique_models")),
                                  cell("63,918", PS, "counts.result_rows", jp(PS, "counts.result_rows"))], until=r"result rows\)")
S(R, r"returns the published\s+numbers to four decimals", [cell("0.9679", RR, "mean_h1", A(Q, "mean_h1")), cell("0.8786", RR, "mean_h4", A(Q, "mean_h4")),
                                                          cell("1.0431", RR, "CORE.mean_h1", A("CORE", "mean_h1")), cell("0.9318", RR, "CORE.mean_h4", A("CORE", "mean_h4"))],
  until=r"for `CORE`")
S(R, r"scores both horizons on the", [cell("54", RR, "oos.n_origins_h1", jp(RR, "oos.n_origins_h1"))], until=r"origins")
S(R, r"On all", [cell("57", f"{XC}/relrmse.csv", "origins h1", lambda: CSV(f"{XC}/relrmse.csv").query("h==1 and model=='VARX_CORE+OIL'").origins.iloc[0]),
                                                  cell("0.9656", f"{XC}/relrmse.csv", "VARX h1 mean_vs_rw", lambda: CSV(f"{XC}/relrmse.csv").query("h==1 and model=='VARX_CORE+OIL'").mean_vs_rw.iloc[0])],
  until=r"0\.9656")
S("VARX-README", r"All growth rates are", [cell("400", VL, "dlog scale", lambda: codenum(VL, r"return (\d+)\.?0? \* np\.log"))], until=r"times")
for pp in ("paper", "study1-paper"):
    S(pp, r"seeded script \(seed", [cell("20260710", RR, "provenance.seed", jp(RR, "provenance.seed"))], until=r"\)")
for pp in ("paper", "varx-paper"):
    S(pp, r"Across", [cell("1{,}632", PS, "counts.spec_rows", jp(PS, "counts.spec_rows"))], until=r"specification cells", after="\\section{Conclusion}")
S(R, r"The Minnesota BVAR uses lambda", [cell("0.1", "src/crosscheck_study1.py", "BVAR_L1", lambda: codenum("src/crosscheck_study1.py", r"BVAR_L1, BVAR_L2, BVAR_L3 = ([\d.]+)")),
                                         cell("1", "src/crosscheck_study1.py", "PRIOR_MEAN policy", lambda: codenum("src/crosscheck_study1.py", r"PRIOR_MEAN = \[[\d., ]*?, ([\d.]+), [\d.]+\]"))])

V = "VARX-README"
S(V, r"the study crosses", [cell("16", VL, "endogenous sets", n_endog), cell("12", VL, "shock subsets", n_shocks)], until=r"shock subsets")
S(V, r"That is", [cell("1,632", PS, "counts.spec_rows", jp(PS, "counts.spec_rows")), cell("1,014", PS, "counts.unique_models", jp(PS, "counts.unique_models"))], until=r"unique models")
S(V, r"cuts RMSE about", [cell("12", PS, "1-best/noshock gdp_1_all", adv12)], until=r"percent")
S(V, r"Across all targets and horizons, only", [cell("2", PS, "dm_n_better_5pct", jp(PS, "dm_n_better_5pct")),
                                                cell("2,826", PS, "dm_tests", jp(PS, "dm_tests"))], until=r"tests favour")
S(V, r"best RMSE", [cell("1.62", PS, "best_by_rule.stance_8_aic", jp(PS, "best_by_rule.stance_8_aic")),
                    cell("2.01", PS, "best_by_rule.stance_8_bic", jp(PS, "best_by_rule.stance_8_bic"))], until=r"BIC\)")
S(V, r"roughly", [cell("45", "src varx_realtime.py", "VLAG_DAYS", lambda: codenum(f"{VARX}/src/varx_realtime.py", r"VLAG_DAYS = (\d+)"))], until=r"days")
S(V, r"leaves the GDP shock advantage almost intact", [
    cell("16.6", RT, "adv_final_pct", lambda: rtg("adv_final_pct")), cell("15.8", RT, "adv_rt_pct", lambda: rtg("adv_rt_pct")),
    cell("7.5", RT, "excovid adv_final_pct", lambda: rtg("adv_final_pct", "excovid")), cell("7.8", RT, "excovid adv_rt_pct", lambda: rtg("adv_rt_pct", "excovid"))])
S(V, r"GDP purging, which degrades", [cell("35", RTC, "all4_gdp_resid degr_pct", lambda: degr("all4_gdp_resid"))], until=r"percent")
S(V, r"The proxy forecast encompasses the identified one in", [
    cell("21", CBC, "beta_p>=0.05 cells", lambda: int((P(CBC).beta_p >= 0.05).sum())), cell("24", CBC, "cells", lambda: len(P(CBC))),
    cell("0.2", CB, "min blend pct", lambda: min(blend())), cell("0.8", CB, "max blend pct", lambda: max(blend()))], until=r"percent worse")
S(V, r"nominal", [cell("68", DN, "naive gdp 1 all all4_raw cov90", lambda: 100 * J(DN)["naive"]["gdp"]["1"]["all"]["all4_raw"]["cov90"])], until=r"percent of outcomes")
S(V, r"restores coverage to", [cell("91", DN, "fair gdp 1 all all4_raw cov90", lambda: 100 * J(DN)["fair"]["gdp"]["1"]["all"]["all4_raw"]["cov90"])], until=r"percent")
S(V, r"walk-forward produces", [cell("1,014", PS, "counts.unique_models", jp(PS, "counts.unique_models")), cell("63,918", PS, "counts.result_rows", jp(PS, "counts.result_rows"))], until=r"result rows")

RRD = "results-README"
S(RRD, r"The phase-2 files and", [cell("127", "results/phase2_summary.json", "n", jp("results/phase2_summary.json", "n"))], until=r"quarters")
S(RRD, r"`reproduce_results.json` uses", [cell("128", RR, "sample_level.n", jp(RR, "sample_level.n"))], until=r"quarters")
S(RRD, r"\(Ridge small:", [cell("0.8792", "results/phase2_modelclass_results.csv", "Ridge:small mean_relRMSE_h1",
                                lambda: CSV("results/phase2_modelclass_results.csv").set_index("model_id").loc["Ridge:small", "mean_relRMSE_h1"]),
                          cell("0.8768", RR, "Ridge:small mean_h1", C("Ridge:small", "mean_h1"))], until=r"here\)")


# ----------------------------------------------------------------------------- evaluating claims
def _num(shown):
    return float(shown.replace("{,}", "").replace(",", ""))


def evaluate(c):
    """Return (ok, message, shown_value, expected_value)."""
    if c["pos"] is None:
        return False, "claimed number not found in the text", None, None
    s, e, shown = c["pos"]
    neg = FLAT[c["doc"]][max(s - 3, 0):s] == "$-$"
    try:
        expected = c["get"]()
    except Exception as ex:  # a claim with no readable source is a failure
        return False, f"source unreadable: {type(ex).__name__}: {ex}", shown, None
    if c["kind"] == "prefix":
        return str(expected).startswith(shown), "version prefix", shown, expected
    val = (-1 if neg else 1) * _num(shown)
    dec = len(shown.split(".")[1]) if "." in shown else 0
    tol = c["tol"] if c["tol"] is not None else 0.5 * 10 ** (-dec) + 1e-9
    if c["kind"] == "lt":
        return float(expected) < val, f"{expected} not < {val}", shown, expected
    if c["kind"] == "gt":
        return float(expected) > val, f"{expected} not > {val}", shown, expected
    return abs(val - float(expected)) <= tol, f"|{val} - {float(expected):.6g}| > {tol:g}", shown, expected


def table():
    rows = []
    for c in CLAIMS:
        ok, msg, shown, exp = evaluate(c)
        rows.append((c["doc"], c["where"][:34], shown, c["src"], c["field"], exp, "ok" if ok else "FAIL " + msg))
    return rows


@pytest.mark.parametrize("c", CLAIMS, ids=[f"{c['doc']}:{c['field']}:{c['pos'][2] if c['pos'] else '?'}@{c['pos'][0] if c['pos'] else '?'}" for c in CLAIMS])
def test_claim(c):
    ok, msg, shown, exp = evaluate(c)
    assert ok, f"[{c['doc']}] {c['where']!r}: shown {shown} vs {c['src']}::{c['field']} = {exp}; {msg}"


# ----------------------------------------------------------------------------- allowlist (explicit, one-word reasons)
ALLOW_RULES = [
    # (reason, doc kind or None, token regex, regex on the text before the token, regex on the text after the token)
    ("year", None, r"(?:19|20)\d\d", r"", r""),
    ("date", None, r"01", r"2006-$", r""),
    ("horizon", None, r"\d+", r"(?:\bh|\\hat\{y\}_h)\s*(?:\{=\}|=|\\in\s*\\\{|-|\\in\s*\{)\s*[\d,\s]*$", r""),
    ("horizon", None, r"\d+", r"\bh=\d+(?:,\s*)?$", r""),
    ("setting", None, r"\d+", r"\b[pn]\s*(?:\{=\}|=|\\in\s*\\\{)\s*[\d,\s]*$", r""),
    ("setting", None, r"\d+", r"\\dots,$", r""),
    ("setting", None, r"2", r"VARX\($", r""),
    ("reference", None, r"\d+", r"(?:Study|Phase|case|lag|Phase-|Table|Fig\.~?)[\s-]*$", r""),
    ("notation", None, r"\d+", r"(?:\^|_)\{?[A-Za-z0-9,\-\s=]*$", r""),
    ("notation", None, r"\d+", r"(?:\(|\[|\\\{)$", r""),
    ("notation", None, r"\d+", r"(?:MA|AR|VAR|\\sum_\{i=)\($", r""),
    ("notation", None, r"\d+", r"\\Delta ?\\ln x_t\$; oil enters as \$$", r""),
    ("scale", None, r"(?:400|100)", r"\$$", r"\\,\\Delta"),
    ("scale", None, r"400", r"", r"times the log"),
    ("threshold", None, r"(?:5|90)", r"", r"\s*(?:percent|%|\\%)"),
    ("threshold", None, r"0\.90", r"nearer $", r""),
    ("unit", None, r"\d+", r"", r"-degree"),
    ("unit", None, r"\d+", r"", r"-point"),
]
EXPLICIT = [  # (doc, text just before the token, reason)
    ("README", "so values below ", "reference"),
    ("results-README", "model-class phase ", "reference"),
    ("results-README", "The phase-", "reference"),
    ("results-README", "from the phase-", "reference"),
]


def classify(doc, s, e, tok):
    kind = DOCS[doc][1]
    pre = FLAT[doc][max(0, s - 40):s]
    post = FLAT[doc][e:e + 30]
    if kind == "md" and RAW[doc][max(0, s - 1):s] == "\n" and RAW[doc][e:e + 1] == ".":
        return "list"
    if kind == "md" and RAW[doc][max(0, s - 4):s].strip() == "" and RAW[doc][e:e + 2] == ". ":
        return "list"
    # first cell of a table row: a label such as "GDP, 1" or "4 &"
    if kind == "tex":
        back = re.split(r"\\\\|\\midrule|\\toprule", FLAT[doc][max(0, s - 200):s])[-1]
        fwd = re.split(r"\\\\", FLAT[doc][e:e + 200])[0]
        if "&" not in back and "&" in fwd:
            return "label"
    for d_, before, reason in EXPLICIT:
        if d_ == doc and FLAT[doc][max(0, s - len(before)):s] == before:
            return reason
    for reason, k, tokre, prere, postre in ALLOW_RULES:
        if k and k != kind:
            continue
        if re.fullmatch(tokre, tok) and re.search(prere + r"\Z" if prere and not prere.endswith("$") else prere or r"", pre if not prere else pre) \
                and (not postre or re.match(postre, post)):
            if prere and not re.search(prere, pre):
                continue
            return reason
    return None


# ----------------------------------------------------------------------------- coverage
def coverage():
    out = {}
    claimed = {}
    for c in CLAIMS:
        if c["pos"]:
            claimed.setdefault(c["doc"], set()).add(c["pos"][0])
    for d in DOCS:
        un, allow = [], {}
        cov = 0
        for s, e, tok in TOK[d]:
            if s in claimed.get(d, set()):
                cov += 1
                continue
            r = classify(d, s, e, tok)
            if r:
                allow.setdefault(r, []).append(tok)
            else:
                un.append((s, tok, FLAT[d][max(0, s - 35):e + 20]))
        out[d] = (cov, allow, un)
    return out


def test_every_number_is_claimed_or_allowlisted():
    bad = []
    for d, (cov, allow, un) in coverage().items():
        for s, tok, ctx in un:
            bad.append(f"{d}: {tok!r} in ...{ctx}...")
    assert not bad, "numbers neither claimed nor allowlisted:\n" + "\n".join(bad)


# ----------------------------------------------------------------------------- settings against the code
def test_varx_settings():
    vl = varx_lab()
    assert vl.TRAIN_END == "2009Q4" and vl.SAMPLE_END == "2026Q1"
    assert vl.LAG_RULES == ["aic", "bic", "hqic"] and vl.MAIN_RULE == "bic"
    assert max(vl.P_GRID) == 4 and min(vl.P_GRID) == 1 and vl.XLAGS == 2
    assert vl.H_REPORT == [1, 4, 8] and vl.COVID_DUMMIES == ["2020Q2", "2020Q3"]
    assert len(vl.enumerate_specs()) == J(PS)["counts"]["cells"] == 544
    assert "400.0" in code(VL) or "400 *" in code(VL) or "400*" in code(VL)
    st = J(PS)["counts"]
    assert (st["sample_start"], st["sample_end"]) == ("1990-01-01", "2026-01-01")


def test_study1_settings():
    for f in ("src/var_horserace.py", "src/reproduce.py", "src/crosscheck_study1.py"):
        assert re.search(r"MAXLAG_VAR\s*(?:,[^=]*)?=\s*4", code(f)), f
        assert 'ic="bic"' in code(f) or "ic='bic'" in code(f), f
    assert re.search(r"\nP = 2\b", code("src/reproduce.py")) and re.search(r"\nP = 2\b", code("src/phase2_modelclass.py"))
    r = J(RR)
    assert r["oos"]["lag_p"] == 2 and r["oos"]["start_frac"] == 0.55
    assert (r["sample_level"]["start"], r["sample_level"]["end"]) == ("1994-04-01", "2026-01-01")
    assert (r["sample_yoy"]["start"], r["sample_yoy"]["n"]) == ("1995-01-01", 125)


def test_lag_design_is_stated_the_same_way_everywhere():
    """BIC is the main rule; AIC and HQIC are robustness checks: code, README and paper."""
    for f in ("src/var_horserace.py", "src/reproduce.py", "src/crosscheck_study1.py", f"{VARX}/src/varx_lab.py"):
        assert re.search(r"BIC is the main", code(f)), f
    assert "BIC is the main lag rule" in FLAT["README"] and "robustness" in FLAT["README"]
    assert "BIC is the main rule" in FLAT["VARX-README"]
    assert "BIC is the main rule" in FLAT["paper"] and "robustness checks" in FLAT["paper"]
    assert "BIC is the main lag rule" in FLAT["varx-paper"] and "robustness checks" in FLAT["varx-paper"]
    assert "chosen by BIC, up to four lags" in FLAT["study1-paper"] and "robustness checks" in FLAT["study1-paper"]
    assert json.loads(json.dumps(J(PS)["p_by_rule"]))["bic"].keys() <= {"1", "2", "3", "4"}


def test_qualitative_claims():
    # no identified-oil DM test rejects at 5 percent (paper, External Identification)
    for t, h, s in CELLS:
        for a in ("oil_raw", "oil_surp", "oil_news"):
            p = J(ID)[t][h][s][a]["dm_p_vs_var"]
            assert not (p == p and p < 0.05)
    # Johansen: no cointegration with an intercept, one vector with no deterministic terms (README)
    jo = {j["det_order"]: j["rank_5pct"] for j in J(f"{XC}/summary.json")["johansen"]}
    assert jo[0] == 0 and jo[-1] == 1
    # every phase-2 estimator and the AR are in the MCS on all four targets; RW is out only for FX (paper)
    m = J(RR)["phaseC_MCS"]
    for tgt in ("GDP", "INF", "POLICY"):
        assert "RW" in m[tgt]
    assert "RW" not in m["FX"] and all(len(m[t]) >= 12 for t in m)
    # every log-score difference under the fair density is insignificant (paper)
    for t in ("gdp", "inf"):
        for hh in J(DN)["fair"][t]:
            for sm in J(DN)["fair"][t][hh]:
                for arm, v in J(DN)["fair"][t][hh][sm].items():
                    p = v["ls_dm_p_vs_var"]
                    assert not (p == p and p < 0.05), (t, hh, sm, arm, p)


def test_lag_rule_facts_in_the_paper():
    """BIC selects p=2 for every headline winner; AIC spreads over p 2 to 4; HQIC picks p=2 almost everywhere."""
    st = J(PS)
    assert all(v["best"]["p"] == 2 for k, v in st.items() if re.match(r"(gdp|inf|stance|fx)_\d_(all|excovid)$", k))
    assert set(st["p_by_rule"]["aic"]) == {"2", "3", "4"} and set(st["p_by_rule"]["bic"]) == {"1", "2"}
    assert st["p_by_rule"]["hqic"]["2"] >= 0.99 * sum(st["p_by_rule"]["hqic"].values())


def test_no_study2_in_text():
    for d in DOCS:
        assert "ExoVar" not in RAW[d] and "study2" not in RAW[d].lower()


def test_claim_count_report():
    cov = coverage()
    n_claims = sum(1 for c in CLAIMS if c["pos"])
    n_allow = sum(len(t) for d in cov.values() for t in d[1].values())
    print(f"claims {len(CLAIMS)} (positioned {n_claims}); allowlisted {n_allow}")


if __name__ == "__main__":
    print(f"{'location':14} {'claim text':34} {'shown':>9} {'source file::field':80} {'expected':>12}  status")
    for r in table():
        print(f"{r[0]:14} {r[1]:34} {str(r[2]):>9} {(r[3] + '::' + r[4])[:80]:80} {str(round(r[5], 5) if isinstance(r[5], float) else r[5]):>12}  {r[6]}")
    cov = coverage()
    tc = sum(v[0] for v in cov.values())
    ta = sum(len(t) for v in cov.values() for t in v[1].values())
    print(f"\nnumbers covered by a claim row: {tc}; allowlisted: {ta}; uncovered: {sum(len(v[2]) for v in cov.values())}")
    for d, (c_, a_, u_) in cov.items():
        print(f"  {d}: covered {c_}, allowlisted {sum(len(x) for x in a_.values())} {({k: len(x) for k, x in a_.items()})}, uncovered {len(u_)}")


def test_headline_none_significantly_better_for_gdp_h1_and_rejections_are_mostly_worse():
    """README/paper say none of the GDP h=1 specs beats the core VAR significantly, and that the 3.8% of
    two-sided rejections are almost all the shocked model being worse (dm_stat > 0)."""
    d = J(PS)
    assert d["gdp1_vs_core"]["n_sig_better_5pct"] == 0
    assert abs(d["gdp1_vs_core"]["share_below_core"] - 2 / 3) < 0.02   # "two thirds" in the READMEs
    assert d["dm_worse_share_5pct"] > d["dm_better_share_5pct"]        # "most significant results go the other way"
    assert d["dm_worse_share_5pct"] > 0.95 * d["dm_sig_share_5pct"]
    assert abs(d["dm_sig_share_5pct"] - d["dm_better_share_5pct"] - d["dm_worse_share_5pct"]) < 0.002
