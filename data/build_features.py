"""Features (year t) and distress labels (year t+1).

Input : panel_items.csv (from build_panel.py)
Output: dataset.csv — one row per firm-year t with features from t and labels from t+1.

Label (main): Article 141 of the Iranian Commercial Code — accumulated losses >= 50% of registered capital,
              measured on the parent-only balance sheet at fiscal year-end t+1.
Alternative labels: negative equity at t+1; net loss in both t and t+1 (two consecutive loss years).
Leakage control: the ratio accumulated-loss/capital and the A141 status at t are NOT features
(the status at t is kept as `a141_t` only for sample selection: onset sample = a141_t == 0).
"""
import numpy as np, pandas as pd

df = pd.read_csv("panel_items_merged.csv")
# sanity: placeholder balance sheets (e.g. total assets = 1, capital = 0) are treated as missing
BS = ["cash","st_inv","trade_rec","other_rec","inventory","cur_assets","ppe","total_assets","trade_pay","other_pay",
      "st_debt","cur_liab","lt_debt","total_liab","capital","retained","equity"]
bad = (df.total_assets < 100) | (df.capital <= 0)
df.loc[bad, [c for c in BS if c in df.columns]] = np.nan
df = df.sort_values(["symbol", "fy"]).reset_index(drop=True)

def sdiv(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((b != 0) & np.isfinite(b), a / b, np.nan)
    return r

f = pd.DataFrame({"symbol": df.symbol, "fy": df.fy, "year_end": df.year_end, "industry": df.industry,
                  "market_state": df.market_state, "filled_from_next": df.filled_from_next,
                  "filled_from_later": df.filled_from_later, "bs_from_interim": df.bs_from_interim})
TA, TL, CA, CL = df.total_assets, df.total_liab, df.cur_assets, df.cur_liab
debt = df[["st_debt", "lt_debt"]].sum(axis=1, min_count=1)
rev = df.revenue
# liquidity
f["current_ratio"] = sdiv(CA, CL)
f["quick_ratio"] = sdiv(CA - df.inventory.fillna(0), CL)
f["cash_ta"] = sdiv(df.cash, TA)
f["wc_ta"] = sdiv(CA - CL, TA)
# leverage / solvency
f["tl_ta"] = sdiv(TL, TA)
f["debt_ta"] = sdiv(debt, TA)
f["st_debt_ta"] = sdiv(df.st_debt, TA)
f["eq_ta"] = sdiv(df.equity, TA)
f["fincost_rev"] = sdiv(df.fin_cost.abs(), rev)
f["int_cover"] = sdiv(df.op_profit, df.fin_cost.abs())
# profitability
f["roa"] = sdiv(df.net_income, TA)
f["ebit_ta"] = sdiv(df.op_profit, TA)
f["gross_margin"] = sdiv(df.gross_profit, rev)
f["op_margin"] = sdiv(df.op_profit, rev)
f["net_margin"] = sdiv(df.net_income, rev)
f["loss_t"] = (df.net_income < 0).astype(float).where(df.net_income.notna())
f["re_ta"] = sdiv(df.retained, TA)          # Altman X2 — extended set only (close to the label)
# activity
f["rev_ta"] = sdiv(rev, TA)
f["inv_ta"] = sdiv(df.inventory, TA)
f["rec_ta"] = sdiv(df[["trade_rec", "other_rec"]].sum(axis=1, min_count=1), TA)
f["pay_ta"] = sdiv(df[["trade_pay", "other_pay"]].sum(axis=1, min_count=1), TA)
f["ppe_ta"] = sdiv(df.ppe, TA)
# cash flow
f["cfo_ta"] = sdiv(df.cfo, TA)
f["cfo_tl"] = sdiv(df.cfo, TL)
f["capex_ta"] = sdiv(df.capex.abs(), TA)
# size and growth (nominal; deflate in modelling or rely on year effects)
f["log_ta"] = np.log(TA.where(TA > 0))
f["log_rev"] = np.log(rev.where(rev > 0))
g = df.groupby("symbol")
prev_fy = g.fy.shift(1)
consec = (df.fy - prev_fy) == 1
f["rev_growth"] = np.where(consec, np.log(rev.where(rev > 0)) - np.log(g.revenue.shift(1).where(lambda x: x > 0)), np.nan)
f["ta_growth"] = np.where(consec, np.log(TA.where(TA > 0)) - np.log(g.total_assets.shift(1).where(lambda x: x > 0)), np.nan)
f["loss_prev"] = np.where(consec, (g.net_income.shift(1) < 0).astype(float), np.nan)

# status at t
cap, ret, eq = df.capital, df.retained, df.equity
a141 = ((ret < 0) & (-ret >= 0.5 * cap)).astype(float).where(ret.notna() & cap.notna() & (cap > 0))
f["a141_t"] = a141
f["neg_equity_t"] = (eq < 0).astype(float).where(eq.notna())

# labels at t+1
nfy = g.fy.shift(-1)
nxt = (nfy - df.fy) == 1
f["y_a141"] = np.where(nxt, a141.groupby(df.symbol).shift(-1), np.nan)
f["y_negeq"] = np.where(nxt, f.neg_equity_t.groupby(df.symbol).shift(-1), np.nan)
nl = (df.net_income < 0).astype(float).where(df.net_income.notna())
f["y_twoloss"] = np.where(nxt, (nl * nl.groupby(df.symbol).shift(-1)), np.nan)
f["capital_t"] = cap; f["total_assets_t"] = TA; f["revenue_t"] = rev   # for SSOA calibration (capacity, price)
f.to_csv("dataset.csv", index=False, encoding="utf-8-sig")

lab = f[f.y_a141.notna()]
print("firm-years with label:", len(lab), "firms:", lab.symbol.nunique())
print("A141 rate (all):", round(lab.y_a141.mean(), 4))
on = lab[lab.a141_t == 0]
print("onset sample:", len(on), "onset rate:", round(on.y_a141.mean(), 4))
print(on.groupby("fy").y_a141.agg(["count", "mean"]).round(3).to_string())
print(f.drop(columns=["symbol", "year_end"]).notna().mean().round(3).to_string())
