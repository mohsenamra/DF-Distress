"""Step 5b — decision quality of accuracy-trained baselines (plug-in and label-based rules).
For each dataset / fold / pool of N suppliers:
  PI-M1 : z*_M1(p_hat)                      (risk-neutral spot-only plug-in)
  PI-M2 : z*_M2(p_hat)                      (full two-stage mean-CVaR plug-in, S scenarios)
  LB    : screen-then-optimise — suppliers with p_hat >= tuned G-mean threshold get p=0.999, others the training base rate
  NAIVE : p = training base rate for all (risk-blind);  ORACLE: z* given realised xi (severity averaged over B draws)
Realised cost = Lemma 1 averaged over B severity draws (common random numbers); regret vs ORACLE.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")

import sys, os, time, numpy as np, pandas as pd
from common import load, rolling_splits, cv_splits
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M1Solver, M2Solver, ScenarioBank, severity_given_xi, realised_cost, m1_decision_to_yqv
import json, zlib
OUT = ROOT + "/results/"
CP = json.load(open(ROOT + "/data/copula_params.json"))
RHO = {"IR": (CP["rho_global_onset"], CP["rho_within_industry_onset"]), "US": (CP["US"]["rho_global"], CP["US"]["rho_within_division"])}
RHO["PL"] = RHO["TW"] = RHO["US"]
S, B, NPOOL, N = 100, 200, int(os.environ.get("NPOOL", 20)), 20

def splits_for(name, d):
    if name == "IR": return list(rolling_splits(d, range(1397, 1404)))
    if name == "US": return list(rolling_splits(d, range(2011, 2018)))
    return [(f"r{r}k{k}", tr, te) for r in range(2) for k, tr, te in cv_splits(d, 5, seed=r)]

def run(name, models=("LR", "LR-CS", "XGB", "XGB-CS", "LGBM", "LGBM-Focal")):
    d, _ = load(name); A = attributes(name)
    sfx = os.environ.get("PREDS", ""); P = pd.read_csv(OUT + f"preds_{name}{sfx}.csv.gz"); P["fold"] = P.fold.astype(str)
    rho0, rhog = RHO[name]; rows = []
    fn = OUT + f"decisions{sfx or '_baselines'}_{name}.csv"
    done = set()
    if os.path.exists(fn):                                   # resume: keep completed folds
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); rows = old.to_dict("records"); done = set(old.fold)
    for fold, tr, te in splits_for(name, d):
        fold = str(fold); base = float(d.loc[tr, "y"].mean())
        if fold in done: continue
        Pf = P[P.fold == fold].set_index(["model", "idx"])
        if Pf.empty: continue                                    # fold not (yet) predicted by these models
        pools = draw_pools(te, A, NPOOL, N, seed=zlib.crc32(f"{name}{fold}".encode()))
        t0 = time.time()
        for k, idx in enumerate(pools):
            rng = np.random.default_rng([zlib.crc32(f"{name}{fold}".encode()), k])
            ins = make_pool_instance(A.loc[idx], rng)
            xi = d.loc[idx, "y"].values.astype(float)
            th_eval = severity_given_xi(ins, xi, B, seed=k + 7)
            bank = ScenarioBank(ins, S, rho0, rhog, seed=k + 11)
            m1 = M1Solver(ins); m2 = M2Solver(ins, bank)
            def cost(yqv): return float(realised_cost(ins, *yqv, th_eval).mean())
            oracle = M2Solver(ins, None, lam=0.0)
            yo = oracle.solve_theta(th_eval)
            C_or = cost(yo)
            res = {"ORACLE": C_or, "NAIVE|M2": cost(m2.solve(np.full(N, base)))}
            for mname in models:
                ph = Pf.loc[mname].loc[idx, "p_cal"].values; thr = Pf.loc[mname].loc[idx, "thr"].values[0]
                res[f"{mname}|PI-M1"] = cost(m1_decision_to_yqv(ins, m1.solve(ph)))
                res[f"{mname}|PI-M2"] = cost(m2.solve(ph))
                if np.isfinite(thr):
                    res[f"{mname}|LB-M2"] = cost(m2.solve(np.where(ph >= thr, 0.999, base)))
            for key, C in res.items():
                mo, rule = (key.split("|") + ["-"])[:2]
                rows.append(dict(dataset=name, fold=fold, pool=k, model=mo, rule=rule, cost=C, oracle=C_or,
                                 regret=C - C_or, nregret=(C - C_or) / C_or, n_distress=int(xi.sum())))
        print(name, fold, len(pools), "pools", round(time.time() - t0, 1), "s", flush=True)
        pd.DataFrame(rows).to_csv(fn, index=False)
    r = pd.DataFrame(rows)
    print(r.groupby(["model", "rule"]).agg(nregret=("nregret", "mean"), cost=("cost", "mean")).round(5).to_string())

if __name__ == "__main__":
    ms = os.environ.get("MODELS")
    for n in sys.argv[1:]:
        run(n, tuple(ms.split(",")) if ms else tuple(pd.read_csv(OUT + f"preds_{n}{os.environ.get('PREDS', '')}.csv.gz", usecols=["model"]).model.unique()))
