"""sensitivity of decision quality to the decision environment (one factor at a time around the base case).
The trained models are NOT retrained: their out-of-sample probabilities are plugged into M2 built under each
alternative environment, i.e. the analysis measures the robustness of the trained policies.
Pools: the first NPOOL (=10) pools of each test fold of evaluate_plugin (same seeds, so the base case reproduces).
Output: results/sens_{name}.csv  (config, fold, pool, model, rule, cost, oracle, nregret). Resumable per (config, fold).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, os, time, zlib, json, numpy as np, pandas as pd
from common import load
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver, ScenarioBank, severity_given_xi, realised_cost
from evaluate_plugin import splits_for, RHO
OUT = ROOT + "/results/"
NPOOL = int(os.environ.get("NPOOL", 10)); S, B = 100, 200

#            name           instance kwargs                     rho multiplier  rho_g=rho_0   lambda
CONFIGS = [("base",          {},                                  1.0, False, 0.3),
           ("kappa_p=0",     {"kappa_p": 0.0},                    1.0, False, 0.3),
           ("kappa_p=0.05",  {"kappa_p": 0.05},                   1.0, False, 0.3),
           ("kappa_p=0.20",  {"kappa_p": 0.20},                   1.0, False, 0.3),
           ("rho=0",         {},                                  0.0, False, 0.3),
           ("rho x2",        {},                                  2.0, False, 0.3),
           ("no-industry",   {},                                  1.0, True,  0.3),
           ("lambda=0",      {},                                  1.0, False, 0.0),
           ("lambda=0.6",    {},                                  1.0, False, 0.6),
           ("capacity=0.25", {"kappa_K": 0.25},                   1.0, False, 0.3),
           ("capacity=0.50", {"kappa_K": 0.50},                   1.0, False, 0.3),
           ("spot e=1.5",    {"e": 1.5},                          1.0, False, 0.3),
           ("spot e=2.2",    {"e": 2.2},                          1.0, False, 0.3)]
MODELS = [("DFM2", "_dfm2"), ("CAL-BASE", "_dfm2"), ("LGBM", ""), ("LGBM-Focal", ""), ("XGB", ""), ("SPO+-L4", "_df")]

def run(name):
    d, _ = load(name); A = attributes(name); rho0, rhog = RHO[name]
    P = {}
    for sfx in ("", "_df", "_dfm2"):
        t = pd.read_csv(OUT + f"preds_{name}{sfx}.csv.gz"); t["fold"] = t.fold.astype(str); P[sfx] = t
    fn = OUT + f"sens_{name}.csv"; rows, done = [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); rows = old.to_dict("records")
        done = set(zip(old.config, old.fold))
    folds_dfm2 = set(P["_dfm2"].fold)
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold not in folds_dfm2: continue
        base = float(d.loc[tr, "y"].mean()); seed = zlib.crc32(f"{name}{fold}".encode())
        pools = draw_pools(te, A, NPOOL, 20, seed=seed)
        lgbm = P[""][(P[""].fold == fold) & (P[""].model == "LGBM")].set_index("idx")
        thr = float(lgbm.thr.iloc[0])
        for cname, ikw, rmul, nogroup, lam in CONFIGS:
            if (cname, fold) in done: continue
            t0 = time.time()
            r0 = min(rho0 * rmul, 0.95); rg = min(rhog * rmul, 0.95)
            if nogroup: rg = r0
            for k, idx in enumerate(pools):
                rng = np.random.default_rng([seed, k]); ins = make_pool_instance(A.loc[idx], rng, **ikw)
                xi = d.loc[idx, "y"].values.astype(float); th = severity_given_xi(ins, xi, B, seed=k + 7)
                m2 = M2Solver(ins, ScenarioBank(ins, S, r0, rg, seed=k + 11), lam=lam)
                Co = realised_cost(ins, *M2Solver(ins, None, lam=0.0).solve_theta(th), th).mean()
                dec = {}
                for m, sfx in MODELS:
                    t = P[sfx]; p = t[(t.fold == fold) & (t.model == m)].set_index("idx").loc[idx, "p_cal"].values
                    dec[(m, "PI-M2")] = p
                pl = lgbm.loc[idx, "p_cal"].values
                dec[("LGBM", "LB-M2")] = np.where(pl >= thr, 0.999, base)
                dec[("NAIVE", "M2")] = np.full(len(idx), base)
                for (m, rule), p in dec.items():
                    C = realised_cost(ins, *m2.solve(p), th).mean()
                    rows.append(dict(dataset=name, config=cname, fold=fold, pool=k, model=m, rule=rule, cost=C, oracle=Co,
                                     nregret=(C - Co) / Co))
            pd.DataFrame(rows).to_csv(fn, index=False)
            print(name, fold, cname, f"{time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    for n in sys.argv[1:]: run(n)
