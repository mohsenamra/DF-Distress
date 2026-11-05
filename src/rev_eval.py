"""Comparator experiments, evaluation part. Scores the plug-in decisions of all methods on the test pools.

For every dataset / fold / test pool (the same pools and seeds as evaluate_plugin.py) and every method:
  cost_mean   realised cost averaged over B severity draws given the realised distress (as in the paper)
  cost_mc     realised mean-CVaR cost over the same draws (lambda = 0.3, alpha = 0.9), the M2 criterion
  oracle_mean / oracle_mc   the ex-post optimal values of the two criteria
  For the main methods the M2 decision is also computed with S = 200 scenarios (cost_mean_S200, cost_mc_S200).
Iran only: an onset sample of test pools drawn from firms that are not in Article 141 status at the decision time.
Output: results/decisions_rev_{name}.csv (resumable by fold).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, os, time, zlib, numpy as np, pandas as pd
from common import load
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver, ScenarioBank, severity_given_xi, realised_cost
from rev_train import mc_cost, LAM, ALPHA
from run_dfm2 import splits_for, RHO

OUT = ROOT + "/results/"
S, B, NPOOL, N = 100, 200, int(os.environ.get("NPOOL", 20)), 20
BASELINES = ("LR", "LR-CS", "XGB", "XGB-CS", "LGBM", "LGBM-Focal")
DFL = ("CE-L4", "CE-L15", "DFD-L15", "SPO+-L4", "SPO+-L15", "PFYL-L4", "DBB-L4")
DFM2 = ("CAL-BASE", "DFM2", "DFM2-max")
REV = ("DFM2-MC", "DFM2-MC-max", "DFM2-noOracle", "CE-cont", "SPO+-warm", "PFYL-warm")
S200 = ("CAL-BASE", "DFM2", "DFM2-MC", "LGBM-Focal", "LGBM", "XGB", "CE-cont")


def load_preds(name):
    out = []
    for sfx in ("", "_df", "_dfm2", "_rev"):
        fn = OUT + f"preds_{name}{sfx}.csv.gz"
        if os.path.exists(fn):
            P = pd.read_csv(fn, usecols=["fold", "model", "idx", "p_cal"]); P["fold"] = P.fold.astype(str); out.append(P)
    P = pd.concat(out)
    return P[P.model.isin(BASELINES + DFL + DFM2 + REV)]


def score_pools(name, d, A, Pf, pools, fold, tag, base, rho0, rhog, seed_key):
    rows = []
    models = [m for m in BASELINES + DFL + DFM2 + REV if m in Pf.index.get_level_values(0)]
    for k, idx in enumerate(pools):
        rng = np.random.default_rng([zlib.crc32(seed_key.encode()), k])
        ins = make_pool_instance(A.loc[idx], rng)
        xi = d.loc[idx, "y"].values.astype(float)
        th = severity_given_xi(ins, xi, B, seed=k + 7)
        bank = ScenarioBank(ins, S, rho0, rhog, seed=k + 11)
        bank2 = ScenarioBank(ins, 2 * S, rho0, rhog, seed=k + 11)
        m2, m2b = M2Solver(ins, bank), M2Solver(ins, bank2)
        def both(z):
            c = realised_cost(ins, *z, th); return float(c.mean()), mc_cost(c)
        om, _ = both(M2Solver(ins, None, lam=0.0).solve_theta(th))
        _, omc = both(M2Solver(ins, None, lam=LAM, alpha=ALPHA).solve_theta(th))
        res = {"NAIVE": (both(m2.solve(np.full(N, base))), None)}
        for m in models:
            ph = Pf.loc[m].loc[idx, "p_cal"].values
            res[m] = (both(m2.solve(ph)), both(m2b.solve(ph)) if m in S200 else None)
        for m, (c1, c2) in res.items():
            rows.append(dict(dataset=name, fold=fold, sample=tag, pool=k, model=m, cost_mean=c1[0], cost_mc=c1[1],
                             oracle_mean=om, oracle_mc=omc, cost_mean_S200=c2[0] if c2 else np.nan,
                             cost_mc_S200=c2[1] if c2 else np.nan, n_distress=int(xi.sum())))
    return rows


def run(name):
    d, _ = load(name); A = attributes(name); rho0, rhog = RHO[name]; P = load_preds(name)
    fn = OUT + f"decisions_rev_{name}.csv"; rows, done = [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); rows = old.to_dict("records"); done = set(old.fold)
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold in done: continue
        Pf = P[P.fold == fold].set_index(["model", "idx"])
        if Pf.empty or "DFM2-MC" not in Pf.index.get_level_values(0): continue
        base = float(d.loc[tr, "y"].mean()); t0 = time.time()
        pools = draw_pools(te, A, NPOOL, N, seed=zlib.crc32(f"{name}{fold}".encode()))
        rows += score_pools(name, d, A, Pf, pools, fold, "all", base, rho0, rhog, f"{name}{fold}")
        if name == "IR":
            onset = te[d.loc[te, "a141_t"].values == 0]
            pools = draw_pools(onset, A, NPOOL, N, seed=zlib.crc32(f"onset{name}{fold}".encode()))
            rows += score_pools(name, d, A, Pf, pools, fold, "onset", base, rho0, rhog, f"onset{name}{fold}")
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(name, fold, "scored", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]: run(n)
