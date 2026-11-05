"""Standardise the three public bankruptcy datasets for DF-Distress.
Sources (mirrors verified against UCI/Kaggle counts):
  Polish  — UCI #365 (Tomczak 2016), 5 horizon files, 43,405 firm-years, 64 ratios   (mirror: github.com/kamileyagci/project-bankruptcy-prediction)
  Taiwan  — UCI #572 (Liang et al. 2016), 6,819 firms, 95 ratios, TEJ 1999–2009         (mirror: github.com/FedeCana00/Bankruptcy-prediction)
  US      — Lombardo et al. (2022, Future Internet) NYSE/NASDAQ 1999–2018, 78,682 firm-years (github.com/sowide/bankruptcy_dataset)
Output: public/std_{name}.csv with columns: dataset, firm, year, group, y, <features>
"""
import numpy as np, pandas as pd
from scipy.io import arff

P = "public/"
def sdiv(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((b != 0) & np.isfinite(b), a / b, np.nan)

# ---------- US (panel with industry and year)
a = pd.read_csv(P + "american_bankruptcy.csv").sort_values(["company_name", "fyear"])
names = dict(X1="ca", X2="cogs", X3="da", X4="ebitda", X5="inv", X6="ni", X7="rec", X8="mv", X9="sales", X10="ta",
             X11="ltd", X12="ebit", X13="gp", X14="cl", X15="re", X16="rev", X17="tl", X18="opex")
a = a.rename(columns=names)
last = a.groupby("company_name").fyear.transform("max")
a["y"] = ((a.status_label == "failed") & (a.fyear == last)).astype(int)   # fiscal year before the Chapter 7/11 filing
us = pd.DataFrame({"dataset": "US", "firm": a.company_name, "year": a.fyear.astype(int), "group": a.Division, "y": a.y})
us["current_ratio"] = sdiv(a.ca, a.cl); us["wc_ta"] = sdiv(a.ca - a.cl, a.ta); us["tl_ta"] = sdiv(a.tl, a.ta)
us["ltd_ta"] = sdiv(a.ltd, a.ta); us["roa"] = sdiv(a.ni, a.ta); us["ebit_ta"] = sdiv(a.ebit, a.ta)
us["ebitda_ta"] = sdiv(a.ebitda, a.ta); us["gross_margin"] = sdiv(a.gp, a.rev); us["net_margin"] = sdiv(a.ni, a.rev)
us["rev_ta"] = sdiv(a.rev, a.ta); us["inv_ta"] = sdiv(a.inv, a.ta); us["rec_ta"] = sdiv(a.rec, a.ta)
us["mv_tl"] = sdiv(a.mv, a.tl); us["opex_rev"] = sdiv(a.opex, a.rev); us["loss_t"] = (a.ni < 0).astype(float)
us["log_ta"] = np.log(a.ta.where(a.ta > 0)); us["log_rev"] = np.log(a.rev.where(a.rev > 0))
g = a.groupby("company_name")
us["rev_growth"] = np.log(a.rev.where(a.rev > 0)) - np.log(g.rev.shift(1).where(lambda s: s > 0))
us["re_ta"] = sdiv(a.re, a.ta)
us.to_csv(P + "std_us.csv", index=False)

# ---------- Polish (5 horizons; firm id/year not released)
frames = []
for h in range(1, 6):
    d, _ = arff.loadarff(P + f"{h}year.arff"); d = pd.DataFrame(d)
    d["y"] = d.pop("class").astype(str).str.strip("b'").astype(int)
    d.insert(0, "horizon", h); frames.append(d)
pl = pd.concat(frames, ignore_index=True)
pl = pl.rename(columns={f"Attr{i}": f"a{i}" for i in range(1, 65)})
pl.insert(0, "dataset", "PL"); pl.insert(1, "firm", np.arange(len(pl))); pl.insert(2, "year", np.nan); pl.insert(3, "group", pl.pop("horizon"))
pl.to_csv(P + "std_polish.csv", index=False)

# ---------- Taiwan (cross-section)
tw = pd.read_csv(P + "taiwan_bankruptcy.csv")
tw.columns = [c.strip() for c in tw.columns]
y = tw.pop("Bankrupt?")
tw.columns = [f"t{i}" for i in range(1, tw.shape[1] + 1)]
tw.insert(0, "dataset", "TW"); tw.insert(1, "firm", np.arange(len(tw))); tw.insert(2, "year", np.nan); tw.insert(3, "group", np.nan); tw.insert(4, "y", y)
tw.to_csv(P + "std_taiwan.csv", index=False)

for n, d in [("US", us), ("PL", pl), ("TW", tw)]:
    print(n, d.shape, "positives", int(d.y.sum()), "rate", round(d.y.mean(), 4),
          "years", (d.year.min(), d.year.max()) if d.year.notna().any() else "-", "groups", d.group.nunique())
