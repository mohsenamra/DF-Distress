"""Tables 4-10 of the paper from the files in results/ (written to results/tables/ as CSV).

Also writes results/decisions_all.csv, the pool-level decision results of all methods on the common test folds,
which make_figures.py uses.
"""
import os, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from common import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); TAB = os.path.join(RES, "tables")
DS = ["IR", "US", "PL", "TW"]
MAIN = ["DFM2|PI-M2", "CAL-BASE|PI-M2", "LGBM-Focal|PI-M2", "LGBM|PI-M2", "XGB|PI-M2", "XGB-CS|PI-M2", "LR|PI-M2",
        "LR-CS|PI-M2", "CE-L15|PI-M2", "DFD-L15|PI-M2", "SPO+-L4|PI-M2", "DBB-L4|PI-M2", "PFYL-L4|PI-M2", "NAIVE|M2"]
BASE6 = ["LR", "LR-CS", "XGB", "XGB-CS", "LGBM", "LGBM-Focal"]


def decisions():
    """All decision files restricted to the folds on which every method was evaluated."""
    frames = []
    for n in DS:
        dfm2 = pd.read_csv(os.path.join(RES, f"decisions_dfm2_{n}.csv")); dfm2["fold"] = dfm2.fold.astype(str)
        folds = set(dfm2.fold)
        for f in sorted(glob.glob(os.path.join(RES, f"decisions_*_{n}.csv"))):
            x = pd.read_csv(f); x["fold"] = x.fold.astype(str); frames.append(x[x.fold.isin(folds)])
    D = pd.concat(frames).drop_duplicates(["dataset", "fold", "pool", "model", "rule"], keep="last")
    D["key"] = D.model + "|" + D.rule
    D.to_csv(os.path.join(RES, "decisions_all.csv"), index=False)
    return D


def cvar(x, a=0.9):
    x = np.sort(np.asarray(x)); return x[int(np.ceil(a * len(x))) - 1:].mean()


def table4():
    rows = []
    for n in DS:
        P = []
        for sfx in ("", "_df", "_dfm2"):
            t = pd.read_csv(os.path.join(RES, f"preds_{n}{sfx}.csv.gz")); t["fold"] = t.fold.astype(str); P.append(t)
        P = pd.concat(P); P = P[P.fold.isin(set(P[P.model == "DFM2"].fold))]
        for (m, f), g in P.groupby(["model", "fold"]):
            if g.y.sum() == 0: continue
            rows.append(dict(dataset=n, model=m, fold=f, **metrics(g.y.values, g.p_cal.values, 0.5)))
    M = pd.DataFrame(rows)
    keep = [k.split("|")[0] for k in MAIN if not k.startswith("NAIVE")]
    T = M[M.model.isin(keep)].groupby(["model", "dataset"])[["auc", "prauc", "brier"]].mean().unstack()
    out = T["auc"][DS].loc[keep].add_prefix("AUC_")
    out["AUC_avg"] = out.mean(axis=1)
    out["PRAUC_avg"] = T["prauc"].loc[keep].mean(axis=1); out["Brier_avg"] = T["brier"].loc[keep].mean(axis=1)
    return out.round(4)


def table5(D):
    x = (D[D.key.isin(MAIN)].groupby(["key", "dataset"]).nregret.mean() * 100).unstack()[DS].loc[MAIN]
    x["avg"] = x.mean(axis=1)
    x["max_excess_over_best_pct"] = ((x[DS] / x[DS].min() - 1) * 100).max(axis=1)
    blocks = (D[D.key.isin(MAIN)].groupby(["dataset", "fold", "key"]).nregret.mean().unstack()[MAIN])
    ranks = np.apply_along_axis(rankdata, 1, blocks.values)
    x["avg_rank"] = ranks.mean(axis=0)
    p = friedmanchisquare(*blocks.values.T).pvalue
    return x.round(2), p


def table6(D):
    P = D.pivot_table(index=["dataset", "fold", "pool"], columns="key", values="nregret")
    rows = []
    for k in MAIN:
        r = dict(key=k, cvar90_pct=cvar(P[k].dropna()) * 100)
        if k != "DFM2|PI-M2":
            d = (P["DFM2|PI-M2"] - P[k]).dropna()
            r.update(reduction_pct=-d.mean() / P[k].mean() * 100, better_pct=(d < 0).mean() * 100,
                     worse_pct=(d > 0).mean() * 100, wilcoxon_p=wilcoxon(d).pvalue)
        rows.append(r)
    return pd.DataFrame(rows).set_index("key").round(4)


def table7(D):
    x = D[D.model.isin(BASE6) & D.rule.isin(["LB-M2", "PI-M1", "PI-M2"])]
    t = (x.groupby(["rule", "dataset"]).nregret.mean() * 100).unstack()[DS]; t["avg"] = t.mean(axis=1)
    return t.round(2)


def table8(D):
    keys = ["DFM2|PI-M2", "CAL-BASE|PI-M2", "DFM2-max|PI-M2", "DFM2-noCal|PI-M2", "DFM2-cold|PI-M2", "DFD-L15|PI-M2"]
    t = (D[D.key.isin(keys)].groupby(["key", "dataset"]).nregret.mean() * 100).unstack()[DS].loc[keys]
    t["avg"] = t.mean(axis=1)
    return t.round(2)


def table9():
    A = pd.read_csv(os.path.join(RES, "saa_IR.csv"))
    ref = A.groupby(["fold", "pool"]).obj_oos.transform("min")
    A["gap_pct"] = (A.obj_oos - ref) / ref * 100; A["optimism_pct"] = (A.obj_oos - A.v_in) / A.obj_oos * 100
    return A.groupby("S")[["gap_pct", "optimism_pct", "secs", "n_suppliers", "backup"]].mean().round(3)


def table10():
    S = pd.concat([pd.read_csv(os.path.join(RES, f"sens_{n}.csv")) for n in DS]); S["key"] = S.model + "|" + S.rule
    t = (S.groupby(["config", "key"]).nregret.mean() * 100).unstack()
    P = S.pivot_table(index=["config", "dataset", "fold", "pool"], columns="key", values="nregret")
    red, pv = {}, {}
    for c in t.index:
        x = P.loc[c]; d = x["DFM2|PI-M2"] - x["LGBM|PI-M2"]
        red[c] = -d.mean() / x["LGBM|PI-M2"].mean() * 100; pv[c] = wilcoxon(d).pvalue
    t["reduction_vs_LGBM_pct"] = pd.Series(red); t["p_vs_LGBM"] = pd.Series(pv)
    return t.round(4)


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    D = decisions()
    table4().to_csv(os.path.join(TAB, "table4_accuracy.csv"))
    t5, p = table5(D); t5.to_csv(os.path.join(TAB, "table5_regret.csv"))
    table6(D).to_csv(os.path.join(TAB, "table6_pairwise.csv"))
    table7(D).to_csv(os.path.join(TAB, "table7_decision_rules.csv"))
    table8(D).to_csv(os.path.join(TAB, "table8_ablation.csv"))
    table9().to_csv(os.path.join(TAB, "table9_saa.csv"))
    table10().to_csv(os.path.join(TAB, "table10_sensitivity.csv"))
    print(t5.to_string()); print(f"Friedman p = {p:.2e}"); print("tables written to", TAB)
