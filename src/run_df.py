"""Step 6b — train DF-Distress, its ablations and DFL benchmarks for every dataset/fold; write out-of-sample probabilities
in the same format as baselines (results/preds_{name}_df.csv.gz), to be scored by evaluate_plugin.py.
Hyper-parameters are fixed a priori across datasets (no test-set tuning):
  backbone LightGBM, lr 0.05; capacity L4 = 4 leaves / 100 rounds, L15 = 15 leaves / 150 rounds
  dfd: mu0 = 1, mu_inf = 0.5, decay 0.97, eps_rel 0.5, M = 8 perturbations, solve_frac 0.1, N = 20 per pool, 120 pools
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")

import sys, os, time, zlib, numpy as np, pandas as pd
from common import load, rolling_splits, cv_splits
from instances import attributes, make_pool_instance, draw_pools
from dfd import PoolSet, train_df, predict_df
OUT = ROOT + "/results/"
CAP = {"L4": dict(num_leaves=4, rounds=100), "L15": dict(num_leaves=15, rounds=150)}
METHODS = [("CE", "ce", {}), ("DFD", "dfd", dict(mu_inf=0.5)), ("DFD-noStein", "dfd", dict(mu_inf=0.5, stein=False)),
           ("DFD-noW", "dfd", dict(mu_inf=0.5, weights=False)), ("DFD-noCal", "dfd", dict(calib=False)),
           ("SPO+", "spo", {}), ("PFYL", "pfyl", {}), ("DBB", "dbb", {})]
NPOOLS = 120
L15_METHODS = ("CE", "DFD", "SPO+")
ABLATIONS = ("DFD-noStein", "DFD-noW", "DFD-noCal")   # ablations and weak benchmarks only at the low-capacity setting

def splits_for(name, d):
    if name == "IR": return list(rolling_splits(d, range(1397, 1404)))
    if name == "US": return list(rolling_splits(d, range(2011, 2018)))
    return [(f"r{r}k{k}", tr, te) for r in range(1) for k, tr, te in cv_splits(d, 5, seed=r)]   # DF: first CV repeat only (compute)

def train_pools(name, d, A, tr, seed):
    if name in ("IR", "US"):
        yrs = sorted(d.loc[tr, "year"].unique()); per = int(np.ceil(NPOOLS / len(yrs))); out = []
        for yr in yrs:
            out += draw_pools(d.index[(d.year == yr) & d.index.isin(tr)], A, per, 20, seed=seed + int(yr))
        return out
    return draw_pools(tr, A, NPOOLS, 20, seed=seed)

def run(name, caps=("L4", "L15"), methods=None):
    d, feats = load(name); A = attributes(name)
    methods = [m for m in METHODS if methods is None or m[0] in methods]
    frames = []; fn = OUT + f"preds_{name}_df.csv.gz"; done = set()
    if os.path.exists(fn):                                   # resume: keep completed folds
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); frames = [old]; done = set(old.fold)
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold in done: continue
        seed = zlib.crc32(f"{name}{fold}".encode()) % 100000
        pools = train_pools(name, d, A, tr, seed)
        pos = pd.Series(np.arange(len(tr)), index=tr)
        rng = np.random.default_rng(seed)
        P = PoolSet([pos.loc[pl].values for pl in pools], [make_pool_instance(A.loc[pl], rng) for pl in pools],
                    d.loc[tr, "y"].values)
        X, y, Xt = d.loc[tr, feats].values, d.loc[tr, "y"].values, d.loc[te, feats].values
        base = y.mean(); lo = np.log(base / (1 - base))
        for cap in caps:
            for label, meth, kw in methods:
                if cap != "L4" and label not in L15_METHODS: continue
                if name != "IR" and label in ABLATIONS: continue   # ablations on the primary dataset only
                t0 = time.time()
                b, log = train_df(meth, X, y, P, init_score=np.full(len(y), lo), seed=seed, **CAP[cap], **kw)
                pt = predict_df(b, Xt, np.full(len(te), lo))
                frames.append(pd.DataFrame(dict(dataset=name, fold=fold, model=f"{label}-{cap}", idx=te, firm=d.loc[te, "firm"].values,
                                                year=d.loc[te, "year"].values, y=d.loc[te, "y"].values, p_raw=pt, p_cal=pt, thr=np.nan)))
                print(name, fold, cap, label, f"{time.time() - t0:.0f}s",
                      "train_regret_last10", round(np.mean([l["regret"] for l in log[-10:]]), 4) if log else "-", flush=True)
        pd.concat(frames).to_csv(fn, index=False)

if __name__ == "__main__":
    for n in sys.argv[1:]: run(n)
