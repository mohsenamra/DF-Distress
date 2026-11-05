"""Step 6c — DF-Distress-M2 (main method) and its ablations, every dataset/fold; resumable.
Writes results/preds_{name}_dfm2.csv.gz (same format as baselines) to be scored by evaluate_plugin.py (PREDS=_dfm2).
Models written per fold:
  CAL-BASE     cross-fitted, Platt-calibrated LGBM (the base layer; = 0 correction rounds)
  DFM2         base + M2 decision-focused correction, rounds selected on validation pools from {0,25,...,150}
  DFM2-max     same model, all 150 rounds (no validation selection)            [ablation]
  DFM2-noCal   correction trained without the calibration term (mu = 0)       [ablation]
  DFM2-cold    correction trained from the base rate instead of the calibrated base layer (no warm start) [ablation]
Fixed a priori for all datasets: 60 training pools of N=20, 15 validation pools, S=100 scenarios, B=50 severity draws,
eps=0.5, M=8 antithetic perturbations, mu=0.3, solve_frac=0.01, 4 leaves, lr=0.08, 150 rounds.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, os, time, zlib, copy, numpy as np, pandas as pd
from common import load, rolling_splits, cv_splits
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver
from dfm2 import calibrated_base, M2Pool, train_correction, sigmoid, logit
import json
OUT = ROOT + "/results/"
CP = json.load(open(ROOT + "/data/copula_params.json"))
RHO = {"IR": (CP["rho_global_onset"], CP["rho_within_industry_onset"]), "US": (CP["US"]["rho_global"], CP["US"]["rho_within_division"])}
RHO["PL"] = RHO["TW"] = RHO["US"]
NTR, NVAL, RND, LR = 60, 15, 150, 0.08
if os.environ.get("DFM2_SMOKE"): NTR, NVAL, RND = 3, 2, 6            # quick functional test only
CHECK = list(range(0, RND + 1, max(RND // 6, 1)))

def splits_for(name, d):
    if name == "IR": return list(rolling_splits(d, range(1397, 1404)))
    if name == "US": return list(rolling_splits(d, range(2011, 2018)))
    return [(f"r0k{k}", tr, te) for k, tr, te in cv_splits(d, 5, seed=0)]

def fit_val_split(name, d, tr, seed):
    if name in ("IR", "US"):
        last = d.loc[tr, "year"].max(); return tr[d.loc[tr, "year"].values < last], tr[d.loc[tr, "year"].values == last]
    rng = np.random.default_rng(seed); m = rng.random(len(tr)) < 0.2
    return tr[~m], tr[m]

def pools_from(name, d, A, idx, n, seed):
    if name in ("IR", "US"):
        yrs = sorted(d.loc[idx, "year"].unique()); per = int(np.ceil(n / len(yrs))); out = []
        for yr in yrs:
            out += draw_pools(idx[d.loc[idx, "year"].values == yr], A, per, 20, seed=seed + int(yr))
        return out[:max(n, len(out))]
    return draw_pools(idx, A, n, 20, seed=seed)

def run(name):
    d, feats = load(name); A = attributes(name); rho0, rhog = RHO[name]
    fn = OUT + f"preds_{name}_dfm2.csv.gz"; frames, done = [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); frames, done = [old], set(old.fold)
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold in done: continue
        t00 = time.time(); seed = zlib.crc32(f"dfm2{name}{fold}".encode()) % 100000
        X, y, Xt = d.loc[tr, feats].values, d.loc[tr, "y"].values, d.loc[te, feats].values
        s0, s0t = calibrated_base(X, y, Xt, seed=seed)
        pos = pd.Series(np.arange(len(tr)), index=tr); base = y.mean()
        fit_idx, val_idx = fit_val_split(name, d, tr, seed)
        rng = np.random.default_rng(seed)
        def mk(idx, k):
            return M2Pool(make_pool_instance(A.loc[idx], rng), d.loc[idx, "y"].values.astype(float), rho0, rhog, seed=seed + k)
        pools, rows = [], []
        for k, idx in enumerate(pools_from(name, d, A, fit_idx, NTR, seed)):
            P = mk(idx, k); r = pos.loc[idx].values
            P.add(M2Solver(P.ins, None, lam=0.0).solve_theta(P.th_real))
            P.solve(sigmoid(s0[r])); P.solve(np.full(len(r), base)); P.solve(np.clip(2 * sigmoid(s0[r]), 0, 1))
            pools.append(P); rows.append(r)
        vpools = [(mk(idx, 5000 + k), pos.loc[idx].values) for k, idx in enumerate(pools_from(name, d, A, val_idx, NVAL, seed + 1))]
        print(name, fold, "setup", f"{time.time() - t00:.0f}s", len(pools), "train pools", len(vpools), "val pools", flush=True)
        out = {"CAL-BASE": sigmoid(s0t)}
        cold0 = np.full(len(y), logit(base)); cold0t = np.full(len(te), logit(base))
        for label, kw, init, initt in (("DFM2", {}, s0, s0t), ("DFM2-noCal", {"mu": 0.0}, s0, s0t), ("DFM2-cold", {}, cold0, cold0t)):
            t0 = time.time(); P_ = [copy.copy(P) for P in pools]
            for P, Q in zip(P_, pools): P.cache, P.L = list(Q.cache), list(Q.L)       # independent caches per variant
            b, log = train_correction(X, y, init, P_, rows, rounds=RND, lr=LR, solve_frac=0.01, seed=seed, **kw)
            val = {}
            for R in CHECK:
                sv = init + (b.predict(X, num_iteration=R, raw_score=True) if R else 0)
                val[R] = float(np.mean([P.regret_exact(sigmoid(sv[r])) for P, r in vpools]))
            best = min(val, key=val.get)
            out[label] = sigmoid(initt + (b.predict(Xt, num_iteration=best, raw_score=True) if best else 0))
            if label == "DFM2": out["DFM2-max"] = sigmoid(initt + b.predict(Xt, raw_score=True))
            print(name, fold, label, f"{time.time() - t0:.0f}s", "val%", {k: round(v * 100, 3) for k, v in val.items()},
                  "-> rounds", best, "train regret", round(log[0], 4), "->", round(float(np.mean(log[-5:])), 4), flush=True)
        for m, p in out.items():
            frames.append(pd.DataFrame(dict(dataset=name, fold=fold, model=m, idx=te, firm=d.loc[te, "firm"].values,
                                            year=d.loc[te, "year"].values, y=d.loc[te, "y"].values, p_raw=p, p_cal=p, thr=np.nan)))
        pd.concat(frames).to_csv(fn, index=False)
        print(name, fold, "fold done", f"{time.time() - t00:.0f}s", flush=True)

if __name__ == "__main__":
    for n in sys.argv[1:]: run(n)
