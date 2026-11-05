"""Step 4 — data preparation for DF-Distress.
Outputs (data/):
  dataset_model.csv   modelling file (Iran) with real (CPI-deflated) growth/size, winsorised copy of features
  cpi_iran.csv        SCI urban CPI (1400 = 100), annual average
  supplier_pool.csv   firm-years in supply-relevant industries (pool for SSOA instances)
  copula_params.json  Vasicek correlation estimates (global and within-industry)
  tables/desc_*.csv   descriptive statistics
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats, optimize, special

os.makedirs("tables", exist_ok=True)
# --- CPI: Statistical Centre of Iran, urban CPI, annual average, base 1400=100 (amar.org.ir inflation calculator, fetched 2026-09-30)
CPI = {1390: 12.33, 1391: 15.851, 1392: 20.935, 1393: 24.039, 1394: 26.752, 1395: 28.558, 1396: 30.88, 1397: 39.104,
       1398: 52.546, 1399: 71.568, 1400: 100.0, 1401: 144.958, 1402: 203.96, 1403: 270.836, 1404: 405.082}
cpi = pd.DataFrame({"year": list(CPI), "cpi": list(CPI.values())})
cpi["inflation"] = cpi.cpi.pct_change()
cpi.to_csv("cpi_iran.csv", index=False)

d = pd.read_csv("dataset.csv")
# fiscal year label = calendar year of year-end; CPI of the year in which most of the fiscal year falls
fy_month = d.year_end.str[5:7].astype(int)
cpi_year = np.where(fy_month <= 6, d.fy - 1, d.fy)       # year-ends in months 1–6 mostly fall in the previous Persian year
d["cpi"] = pd.Series(cpi_year).map(CPI).values
d = d.sort_values(["symbol", "fy"]).reset_index(drop=True)
prev_cpi = d.groupby("symbol").cpi.shift(1)
d["infl_log"] = np.log(d.cpi / prev_cpi)
d["rev_growth_real"] = d.rev_growth - d.infl_log
d["ta_growth_real"] = d.ta_growth - d.infl_log
d["log_ta_real"] = np.log((d.total_assets_t / d.cpi * 100).where(d.total_assets_t > 0))
d["log_rev_real"] = np.log((d.revenue_t / d.cpi * 100).where(d.revenue_t > 0))

FEATURES = ["current_ratio", "quick_ratio", "cash_ta", "wc_ta", "tl_ta", "debt_ta", "st_debt_ta", "eq_ta", "fincost_rev",
            "int_cover", "roa", "ebit_ta", "gross_margin", "op_margin", "net_margin", "loss_t", "rev_ta", "inv_ta",
            "rec_ta", "pay_ta", "ppe_ta", "cfo_ta", "cfo_tl", "capex_ta", "log_ta_real", "log_rev_real",
            "rev_growth_real", "ta_growth_real", "loss_prev", "a141_t"]
EXTENDED = ["re_ta"]
# winsorised copies (1st/99th percentile by fiscal year) — used for modelling; raw kept for audit
for c in FEATURES + EXTENDED:
    if d[c].dropna().isin([0, 1]).all():
        continue
    lo = d.groupby("fy")[c].transform(lambda s: s.quantile(0.01))
    hi = d.groupby("fy")[c].transform(lambda s: s.quantile(0.99))
    d[c + "_w"] = d[c].clip(lo, hi)

# --- supply-relevant industries (Codal industry-group ids)
SUPPLY_IG = {27: "Basic metals", 28: "Fabricated metal products", 29: "Machinery & equipment", 31: "Electrical machinery",
             34: "Automotive & parts", 25: "Rubber & plastics", 44: "Chemical products", 24: "Chemicals",
             53: "Cement, lime & plaster", 54: "Other non-metallic minerals", 13: "Metal ore mining",
             23: "Refined petroleum", 21: "Paper products", 49: "Tiles & ceramics", 17: "Textiles"}
d["supply_pool"] = d.industry.isin(SUPPLY_IG).astype(int)
d["industry_name"] = d.industry.map(SUPPLY_IG)
lab = d[d.y_a141.notna()].copy()
lab.to_csv("dataset_model.csv", index=False, encoding="utf-8-sig")
pool = lab[lab.supply_pool == 1]
pool.to_csv("supplier_pool.csv", index=False, encoding="utf-8-sig")

# --- descriptive tables
by_year = lab.groupby("fy").agg(firms=("symbol", "nunique"), obs=("symbol", "size"), a141_t=("a141_t", "mean"),
                                y_a141=("y_a141", "mean"), onset_obs=("a141_t", lambda s: (s == 0).sum()))
onset = lab[lab.a141_t == 0].groupby("fy").y_a141.agg(["sum", "mean"]).rename(columns={"sum": "onset_events", "mean": "onset_rate"})
by_year = by_year.join(onset)
by_year["inflation"] = by_year.index.map(lambda y: cpi.set_index("year").inflation.get(y))
by_year.round(4).to_csv("tables/desc_by_year.csv")

rows = []
for c in FEATURES:
    col = c + "_w" if c + "_w" in lab else c
    s = lab[col]; s0 = lab.loc[lab.y_a141 == 0, col].dropna(); s1 = lab.loc[lab.y_a141 == 1, col].dropna()
    p = stats.mannwhitneyu(s0, s1).pvalue if len(s1) > 5 and len(s0) > 5 else np.nan
    rows.append(dict(feature=c, n=s.notna().sum(), mean=s.mean(), sd=s.std(), p25=s.quantile(.25), median=s.median(),
                     p75=s.quantile(.75), mean_healthy=s0.mean(), mean_distressed=s1.mean(), mw_p=p, missing=s.isna().mean()))
pd.DataFrame(rows).round(4).to_csv("tables/desc_features.csv", index=False)
pool.groupby("industry_name").agg(firms=("symbol", "nunique"), obs=("symbol", "size"),
                                  y_a141=("y_a141", "mean")).round(3).sort_values("obs", ascending=False).to_csv("tables/desc_pool_industries.csv")

# --- Vasicek / binomial-mixture MLE of asset correlation
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40); GH_W = GH_W / GH_W.sum()
def negll(params, n, k, pd_bar):
    rho = special.expit(params[0]); c = stats.norm.ppf(pd_bar)
    pm = stats.norm.cdf((c - np.sqrt(rho) * GH_X[None, :]) / np.sqrt(1 - rho))       # conditional PD at factor nodes
    lik = (stats.binom.pmf(k[:, None], n[:, None], pm) * GH_W[None, :]).sum(1)
    return -np.log(np.clip(lik, 1e-300, None)).sum()

def fit_rho(groups):
    n = groups["n"].to_numpy(); k = groups["k"].to_numpy()
    pd_bar = k.sum() / n.sum()
    r = optimize.minimize(negll, x0=[special.logit(0.1)], args=(n, k, pd_bar), method="Nelder-Mead")
    return float(special.expit(r.x[0])), float(pd_bar), int(len(n))

ons = lab[lab.a141_t == 0]
g_year = ons.groupby("fy").y_a141.agg(n="size", k="sum").reset_index()
rho0, pd0, T = fit_rho(g_year)
g_ind = ons[ons.supply_pool == 1].groupby(["industry", "fy"]).y_a141.agg(n="size", k="sum").reset_index()
g_ind = g_ind[g_ind.n >= 5]
rhow, pdw, G = fit_rho(g_ind)
# persistence-inclusive (full sample) for comparison
g_year_full = lab.groupby("fy").y_a141.agg(n="size", k="sum").reset_index()
rho0_full, _, _ = fit_rho(g_year_full)
params = {"rho_global_onset": rho0, "pd_onset": pd0, "years": T,
          "rho_within_industry_onset": rhow, "industry_year_cells": G,
          "rho_global_full_sample": rho0_full,
          "note": "Binomial-Vasicek MLE (Gauss-Hermite, 40 nodes). Onset sample = firms not in Art.141 at t. "
                  "rho_within_industry is the total within-industry correlation (rho_g); global factor share = rho_global."}
json.dump(params, open("copula_params.json", "w"), indent=2)

print(by_year.round(3).to_string())
print(json.dumps(params, indent=2))
print("labelled:", len(lab), "firms", lab.symbol.nunique(), "| pool obs:", len(pool), "pool firms:", pool.symbol.nunique())
print(pd.read_csv("tables/desc_pool_industries.csv").to_string(index=False))
