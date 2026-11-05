"""Summary tables of the decision results: pooled regret by method and dataset, reductions of DF-Distress with
fold-cluster bootstrap intervals, alternative and misspecified decision environments, Friedman ranks and the
evaluation on realised supplier outcomes. Reads the CSV files in results/ and prints the tables.

    python src/summarise_results.py
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DS = ["IR", "US", "PL", "TW"]
METHODS = ["DFM2", "CE-cont", "CAL-BASE", "LGBM-Focal", "LGBM", "XGB", "XGB-CS", "LR", "LR-CS", "CE-L15", "DFD-L15",
           "SPO+-warm", "PFYL-warm", "SPO+-L4", "SPO+-L15", "NAIVE"]
B = 2000


def boot_index(index, seed=1):
    """Fold-cluster bootstrap stratified by dataset: resample folds within each dataset, keep pools of a fold together."""
    rng = np.random.default_rng(seed)
    ds, fo = index.get_level_values(0), index.get_level_values(1)
    groups = [[np.where((ds == d) & (fo == f))[0] for f in sorted(set(fo[ds == d]))] for d in sorted(set(ds))]
    return [np.concatenate([np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))]) for g in groups])
            for _ in range(B)]


def reduction(a, b, idx):
    """1 - mean(a)/mean(b) with a 95% bootstrap interval, in percent."""
    bs = [1 - a[i].mean() / b[i].mean() for i in idx]
    return 100 * (1 - a.mean() / b.mean()), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def main_tables():
    d = pd.concat([pd.read_csv(os.path.join(RES, f"decisions_rev_{s}.csv")) for s in DS])
    d = d[d["sample"] == "all"].copy()
    d["fold"] = d.fold.astype(str)
    d["r"] = 100 * (d.cost_mean - d.oracle_mean) / d.oracle_mean
    p = d.pivot_table(index=["dataset", "fold", "pool"], columns="model", values="r")
    m = [x for x in METHODS if x in p]
    idx = boot_index(p.index)
    rows = []
    for x in m:
        v = p[x].values
        red = reduction(p["DFM2"].values, v, idx) if x != "DFM2" else (np.nan,) * 3
        rows.append(dict(method=x, **{s: p.loc[s, x].mean() for s in DS}, all_pools=v.mean(),
                         tail=np.sort(v)[-len(v) // 10:].mean(), reduction=red[0], lo=red[1], hi=red[2],
                         better=100 * (p["DFM2"].values < v - 1e-9).mean(),
                         worse=100 * (p["DFM2"].values > v + 1e-9).mean()))
    print(f"Normalised regret (%), {len(p)} test pools; reductions of DF-Distress (DFM2) relative to each method")
    print(pd.DataFrame(rows).set_index("method").round(3).to_string(), "\n")
    blocks = p.groupby(level=[0, 1]).mean()[m].dropna()
    ranks = blocks.rank(axis=1).mean().sort_values()
    chi, pv = stats.friedmanchisquare(*[blocks[x] for x in m])
    print(f"Average Friedman ranks over {len(blocks)} dataset-fold blocks (chi2 = {chi:.1f}, p = {pv:.1e})")
    print(ranks.round(2).to_string(), "\n")


def environments():
    e = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "env_*.csv"))])
    e = e[e["sample"] == "all"]
    t = 100 * e.pivot_table(index="env", columns="model", values="nregret", aggfunc="mean")
    print("Alternative and misspecified decision environments: normalised regret (%)")
    print(t.round(3).to_string(), "\n")


def realised():
    fn = os.path.join(RES, "cal_eval_IR.csv")
    if not os.path.exists(fn):
        return
    c = pd.read_csv(fn)
    c = c[c.outcome == "rev"]
    t = 100 * c.pivot_table(index="model", columns="planner", values="nregret")
    print("Realised supplier outcomes (Iran, revenue-based shortfall): normalised regret (%)")
    print(t.sort_values("paper").round(3).to_string(), "\n")


if __name__ == "__main__":
    main_tables()
    environments()
    realised()
