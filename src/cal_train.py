"""Data-calibrated decision environment (Iran): training part. One fold per call; resumable.

Within each rolling fold only information available at the decision time is used:
  1. the realised delivery shortfall of a supplier is theta = clip(1 - R_{t+1}/R_t, 0, 1) with R the CPI-deflated
     revenue of the audited statements; it is known for training firm-years (outcome year <= test year)
  2. the severity model of the planner is calibrated on the training firm-years: one Beta law for suppliers in
     distress at t+1 and one for the others, by the method of moments (replacing the leverage-based law of Section 3)
  3. DFM2-cal: the decision-focused correction is trained with the planner built on the calibrated severity model
     and with the decision loss evaluated on the realised shortfalls of the training pools; the rounds are chosen on
     validation pools by realised regret.  CE-cont-cal: the matched logistic continuation, rounds chosen the same way.
Same seeds, base layer, pools and hyper-parameters as run_dfm2.py.
Output: results/preds_IR_cal.csv.gz, results/cal_params_IR.csv
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, time, zlib, json, copy, numpy as np, pandas as pd, lightgbm as lgb
from common import load
from instances import attributes, make_pool_instance
from ssoa import M2Solver, realised_cost
from dfm2 import calibrated_base, M2Pool, train_correction, sigmoid
from run_dfm2 import splits_for, fit_val_split, pools_from, RHO
from real_eval import realised_shortfall

OUT = ROOT + "/results/"
NTR, NVAL, RND, LR = 60, 15, 150, 0.08
if _os.environ.get("CAL_SMOKE"): NTR, NVAL, RND = 3, 2, 6
CHECK = list(range(0, RND + 1, max(RND // 6, 1)))


def beta_mom(x):
    x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4); m, v = x.mean(), x.var()
    phi = max(m * (1 - m) / max(v, 1e-6) - 1, 0.5)
    return float(m), float(phi)


def calibration(d, theta, rows):
    t = theta[rows]; y = d.loc[rows, "y"].values; ok = np.isfinite(t)
    m1, p1 = beta_mom(t[ok & (y == 1)]); m0, p0 = beta_mom(t[ok & (y == 0)])
    return dict(mu_min=m1, mu_max=m1, phi=p1, delta0=m0, phi0=p0)


class RealPool(M2Pool):
    """M2 pool whose decision loss is the realised cost under the realised shortfalls (one realisation)."""
    def __init__(self, ins, xi_real, theta_real, rho0, rhog, seed, S=100):
        super().__init__(ins, xi_real, rho0, rhog, seed, S, B=1)
        self.th_real = theta_real[None, :]
        self.cache, self.L = [], []
        self.oracle_cost = self._real(M2Solver(ins, None, lam=0.0).solve_theta(self.th_real))


def logistic_objective(preds, data):
    y = data.get_label(); p = sigmoid(preds)
    return p - y, np.maximum(p * (1 - p), 1e-4)


def run(fold_req):
    name = "IR"; d, feats = load(name); A = attributes(name); rho0, rhog = RHO[name]
    TH = realised_shortfall(d)["rev"].values
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold != fold_req: continue
        fn = OUT + f"preds_IR_cal_{fold}.csv.gz"
        if _os.path.exists(fn): print("done", fold); return
        t00 = time.time(); seed = zlib.crc32(f"dfm2{name}{fold}".encode()) % 100000
        X, y, Xt = d.loc[tr, feats].values, d.loc[tr, "y"].values, d.loc[te, feats].values
        s0, s0t = calibrated_base(X, y, Xt, seed=seed)
        pos = pd.Series(np.arange(len(tr)), index=tr); base = y.mean()
        fit_idx, val_idx = fit_val_split(name, d, tr, seed)
        cal = calibration(d, TH, tr)
        theta = pd.Series(TH, index=d.index)
        fillv = {1: cal["mu_min"], 0: cal["delta0"]}
        rng = np.random.default_rng(seed)
        def mk(idx, k):
            ins = make_pool_instance(A.loc[idx], rng, **cal)
            xi = d.loc[idx, "y"].values.astype(float)
            th = theta.loc[idx].values.copy(); miss = ~np.isfinite(th)
            th[miss] = [fillv[int(v)] for v in xi[miss]]
            return RealPool(ins, xi, th, rho0, rhog, seed=seed + k)
        pools, rows = [], []
        for k, idx in enumerate(pools_from(name, d, A, fit_idx, NTR, seed)):
            P = mk(idx, k); r = pos.loc[idx].values
            P.add(M2Solver(P.ins, None, lam=0.0).solve_theta(P.th_real))
            P.solve(sigmoid(s0[r])); P.solve(np.full(len(r), base)); P.solve(np.clip(2 * sigmoid(s0[r]), 0, 1))
            pools.append(P); rows.append(r)
        vpools = [(mk(idx, 5000 + k), pos.loc[idx].values)
                  for k, idx in enumerate(pools_from(name, d, A, val_idx, NVAL, seed + 1))]
        print(fold, "setup", f"{time.time() - t00:.0f}s", cal, flush=True)

        def select(pr):
            val = {}
            for R in CHECK:
                sv = s0 + (pr(X, R) if R else 0)
                val[R] = float(np.mean([P.regret_exact(sigmoid(sv[r])) for P, r in vpools]))
            return min(val, key=val.get), val

        out, logs = {}, []
        P_ = [copy.copy(P) for P in pools]
        for P, Q in zip(P_, pools): P.cache, P.L = list(Q.cache), list(Q.L)
        b, _ = train_correction(X, y, s0, P_, rows, rounds=RND, lr=LR, solve_frac=0.01, seed=seed)
        pr = lambda Xm, R: b.predict(Xm, num_iteration=R, raw_score=True)
        best, val = select(pr)
        out["DFM2-cal"] = sigmoid(s0t + (pr(Xt, best) if best else 0)); logs.append(("DFM2-cal", best, val))
        params = dict(objective=logistic_objective, num_leaves=4, learning_rate=LR, bagging_fraction=0.8, bagging_freq=1,
                      feature_fraction=0.8, min_data_in_leaf=20, lambda_l2=1.0, max_delta_step=1.0, verbose=-1,
                      num_threads=1, seed=seed)
        bc = lgb.train(params, lgb.Dataset(X, y, init_score=s0, free_raw_data=False), num_boost_round=RND)
        prc = lambda Xm, R: bc.predict(Xm, num_iteration=R, raw_score=True)
        best, val = select(prc)
        out["CE-cont-cal"] = sigmoid(s0t + (prc(Xt, best) if best else 0)); logs.append(("CE-cont-cal", best, val))
        frames = [pd.DataFrame(dict(dataset=name, fold=fold, model=m, idx=te, firm=d.loc[te, "firm"].values,
                                    year=d.loc[te, "year"].values, y=d.loc[te, "y"].values, p_raw=p, p_cal=p, thr=np.nan))
                  for m, p in out.items()]
        pd.concat(frames).to_csv(fn, index=False)
        pd.DataFrame([dict(fold=fold, model=m, rounds=r, val=json.dumps({k: round(v * 100, 4) for k, v in vv.items()}),
                           seconds=round(time.time() - t00), **cal) for m, r, vv in logs]).to_csv(
            OUT + f"cal_params_IR_{fold}.csv", index=False)
        print(fold, "done", f"{time.time() - t00:.0f}s", [(m, r) for m, r, _ in logs], flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
