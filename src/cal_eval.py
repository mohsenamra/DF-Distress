"""Data-calibrated decision environment (Iran): evaluation on realised outcomes.

Test pools as in rev_eval.py / real_eval.py. Every method's probabilities are plugged into two planners:
  paper       M2 with the severity model of Section 3
  calibrated  M2 with the severity model calibrated on the fold's training firm-years (cal_train.py)
and the decisions are scored on the realised shortfalls (revenue-based, and production-volume based where the
Codal monthly report exists). Output: results/cal_eval_IR.csv
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import glob, time, zlib, numpy as np, pandas as pd
from common import load
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver, ScenarioBank, realised_cost
from run_dfm2 import splits_for, RHO
from real_eval import realised_shortfall, MODELS
from cal_train import calibration

OUT = ROOT + "/results/"
S, NPOOL, N = 100, int(_os.environ.get("NPOOL", 20)), 20


def run():
    name = "IR"; d, _ = load(name); A = attributes(name); rho0, rhog = RHO[name]
    TH = realised_shortfall(d)
    P = {}
    for _, sfx in MODELS:
        if sfx not in P:
            t = pd.read_csv(OUT + f"preds_{name}{sfx}.csv.gz", usecols=["fold", "model", "idx", "p_cal"])
            t["fold"] = t.fold.astype(str); P[sfx] = t
    cal = pd.concat([pd.read_csv(f, usecols=["fold", "model", "idx", "p_cal"]) for f in glob.glob(OUT + "preds_IR_cal_*.csv.gz")])
    cal["fold"] = cal.fold.astype(str)
    models = list(MODELS) + [("DFM2-cal", "cal"), ("CE-cont-cal", "cal")]; P["cal"] = cal
    rows = []
    for fold, tr, te in splits_for(name, d):
        fold = str(fold); t0 = time.time(); key = zlib.crc32(f"{name}{fold}".encode())
        if fold not in set(cal.fold): continue
        base = float(d.loc[tr, "y"].mean()); cp = calibration(d, TH["rev"].values, tr)
        probs = {m: P[s][(P[s].fold == fold) & (P[s].model == m)].set_index("idx").p_cal for m, s in models}
        fill = {o: {y: np.nanmean(TH.loc[te][d.loc[te, "y"] == y][o]) for y in (0, 1)} for o in ("rev", "vol")}
        for k, idx in enumerate(draw_pools(te, A, NPOOL, N, seed=key)):
            xi = d.loc[idx, "y"].values
            n_real = int(np.isfinite(TH.loc[idx, "rev"].values).sum())
            if n_real < N - 2: continue
            ins = make_pool_instance(A.loc[idx], np.random.default_rng([key, k]))
            insc = make_pool_instance(A.loc[idx], np.random.default_rng([key, k]), **cp)
            for planner, I in (("paper", ins), ("calibrated", insc)):
                m2 = M2Solver(I, ScenarioBank(I, S, rho0, rhog, seed=k + 11))
                dec = {"NAIVE": m2.solve(np.full(N, base))}
                for m in probs: dec[m] = m2.solve(probs[m].loc[idx].values)
                for o in ("rev", "vol"):
                    th = TH.loc[idx, o].values.copy(); miss = ~np.isfinite(th)
                    th[miss] = [fill[o][int(v)] for v in xi[miss]]; th = th[None, :]
                    orc = float(realised_cost(ins, *M2Solver(ins, None, lam=0.0).solve_theta(th), th).mean())
                    for m, z in dec.items():
                        c = float(realised_cost(ins, *z, th).mean())
                        rows.append(dict(fold=fold, pool=k, planner=planner, outcome=o, model=m, nregret=(c - orc) / orc,
                                         n_distress=int(xi.sum()), n_vol=int(TH.loc[idx, "has_vol"].sum())))
        pd.DataFrame(rows).to_csv(OUT + "cal_eval_IR.csv", index=False)
        print(name, fold, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
