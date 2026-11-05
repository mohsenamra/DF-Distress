"""SAA scenario-size study (Proposition 3): in-sample optimum v_S, out-of-sample objective of z_S on an
independent bank of 5,000 scenarios, optimality-gap estimate and solve time, for S in {25, 50, 100, 200, 400}.
Instances: 10 test pools from each of three Iran folds, probabilities from DF-Distress-M2. Output: results/saa_IR.csv
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import os, time, zlib, numpy as np, pandas as pd
from common import load
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver, ScenarioBank
from dfm2 import FastBank, realised_cost_batch, mean_cvar_rows
from evaluate_plugin import splits_for, RHO
OUT = ROOT + "/results/"; SIZES = [25, 50, 100, 200, 400]; REPS = 3

def run(name="IR", folds=("1399", "1401", "1403")):
    d, _ = load(name); A = attributes(name); rho0, rhog = RHO[name]
    P = pd.read_csv(OUT + f"preds_{name}_dfm2.csv.gz"); P["fold"] = P.fold.astype(str); P = P[P.model == "DFM2"]
    fn = OUT + f"saa_{name}.csv"; rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(r["fold"], r["pool"]) for r in rows}
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold not in folds: continue
        seed = zlib.crc32(f"{name}{fold}".encode())
        for k, idx in enumerate(draw_pools(te, A, 10, 20, seed=seed)):
            if (fold, k) in done: continue
            ins = make_pool_instance(A.loc[idx], np.random.default_rng([seed, k]))
            p = P[P.fold == fold].set_index("idx").loc[idx, "p_cal"].values
            big = FastBank(ins, 5000, rho0, rhog, seed=999)
            th_big = big.draw(p)[1]
            for S in SIZES:
                for r in range(REPS):
                    t0 = time.time()
                    y, q, v, vS = M2Solver(ins, ScenarioBank(ins, S, rho0, rhog, seed=1000 * r + S)).solve(p, return_obj=True)
                    secs = time.time() - t0
                    oos = float(mean_cvar_rows(realised_cost_batch(ins, [(y, q, v)], th_big))[0])
                    rows.append(dict(dataset=name, fold=fold, pool=k, S=S, rep=r, v_in=vS, obj_oos=oos, secs=secs,
                                     n_suppliers=int(y.sum()), backup=float(v.sum())))
            pd.DataFrame(rows).to_csv(fn, index=False)
            print(name, fold, "pool", k, "done", flush=True)

if __name__ == "__main__":
    run()
