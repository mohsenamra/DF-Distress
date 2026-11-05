"""Robustness experiments, training part. One job = one dataset and one configuration; resumable by fold.

Every configuration changes one element of the main DFM2 run (run_dfm2.py) and keeps the rest, including seeds,
base layer, training and validation pools:
  eps025, eps1     smoothing scale eps = 0.25 / 1.0                      (main: 0.5)
  mu01, mu1        calibration weight mu = 0.1 / 1.0                     (main: 0.3)
  m16              M = 16 antithetic perturbations                       (main: 8)
  l8               8 leaves per correction tree                          (main: 4)
  rnd300           300 correction rounds, selection grid {0, 50, ..., 300} (main: 150, grid of 25)
  ntr20, ntr120    20 / 120 training pools (learning curve)              (main: 60)
  m4               M = 4 antithetic perturbations
  ntr40            40 training pools (learning curve)
  strict           strictly separated pipeline: validation firm-years removed from every fitting step (base learner,
                   Platt map and correction are fitted on the fitting part only, the rounds are chosen on pools of the
                   held-out part); cross-fitting folds of the base learner grouped by firm; for the public datasets the
                   1/99% winsorisation bounds estimated on the fitting part of each training window only
  lag1   (IR)      features taken from the statements of year t-1 (one more year of reporting lag)
  negeq  (IR)      alternative distress label: negative book equity at t+1
For strict, lag1 and negeq the calibrated base (CAL-BASE) and the matched logistic continuation (CE-cont) are
refitted under the same conditions so that the comparison stays within the configuration.
Output: results/preds_{name}_s3_{config}.csv.gz and results/s3_train_{name}_{config}.csv (rounds, validation curve, timing).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, os, time, zlib, json, copy, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from common import load
from instances import attributes, make_pool_instance
from ssoa import M2Solver
from dfm2 import LGB_BASE, calibrated_base, M2Pool, train_correction, sigmoid
from run_dfm2 import splits_for, fit_val_split, pools_from, RHO

OUT = ROOT + "/results/"
BASE = dict(eps=0.5, mu=0.3, M=8, leaves=4, rounds=150, ntr=60, nval=15, lr=0.08)
CONFIGS = {"eps025": dict(eps=0.25), "eps1": dict(eps=1.0), "mu01": dict(mu=0.1), "mu1": dict(mu=1.0),
           "m16": dict(M=16), "l8": dict(leaves=8), "rnd300": dict(rounds=300), "ntr20": dict(ntr=20),
           "ntr120": dict(ntr=120), "m4": dict(M=4), "ntr40": dict(ntr=40),
           "strict": dict(strict=True), "lag1": dict(lag=True), "negeq": dict(label="negeq")}
WITH_CONTROLS = ("strict", "lag1", "negeq")
if os.environ.get("S3_SMOKE"): BASE.update(rounds=6, ntr=3, nval=2)


def load_raw_public(name):
    """As common.load for the public datasets, without the full-sample winsorisation."""
    f = {"US": "std_us.csv", "PL": "std_polish.csv", "TW": "std_taiwan.csv"}[name]
    fp = ROOT + "/data/public/" + f
    d = pd.read_csv(fp if os.path.exists(fp) else fp + ".gz")
    feats = [c for c in d.columns if c not in ("dataset", "firm", "year", "group", "y", "re_ta")]
    d = d[d.y.notna()].reset_index(drop=True); d["y"] = d.y.astype(int)
    return d, feats


def winsorise_in_window(d, feats, rows):
    d = d.copy()
    for col in feats:
        if d.loc[rows, col].nunique() > 2:
            lo, hi = d.loc[rows, col].quantile([0.01, 0.99]); d[col] = d[col].clip(lo, hi)
    return d


def calibrated_base_grouped(X, y, Xte, groups, seed=0, k=5):
    """calibrated_base with the cross-fitting folds grouped by firm."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups):
        b = lgb.train(dict(LGB_BASE, seed=seed), lgb.Dataset(X[tr], y[tr]), 300)
        oof[va] = b.predict(X[va], raw_score=True)
    pl = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)
    full = lgb.train(dict(LGB_BASE, seed=seed), lgb.Dataset(X, y), 300).predict(Xte, raw_score=True)
    a, c = pl.coef_[0, 0], pl.intercept_[0]
    return a * oof + c, a * full + c


def load_variant(name, cfg):
    d, feats = load_raw_public(name) if cfg.get("strict") and name != "IR" else load(name)
    if cfg.get("lag") or cfg.get("label") == "negeq":
        d = d.sort_values(["firm", "year"])
        g = d.groupby("firm")
        consecutive_prev = (d.year - g.year.shift(1)) == 1
        consecutive_next = (g.year.shift(-1) - d.year) == 1
        if cfg.get("lag"):
            lagged = g[feats].shift(1)
            d[feats] = lagged.where(consecutive_prev, np.nan)
        if cfg.get("label") == "negeq":
            nxt = g.eq_ta_w.shift(-1).where(consecutive_next, np.nan)
            d["y"] = (nxt < 0).astype(float).where(nxt.notna(), np.nan)
            d = d[d.y.notna()]
            d["y"] = d.y.astype(int)
        d = d.sort_index()
    return d, feats


def logistic_objective(preds, data):
    y = data.get_label(); p = sigmoid(preds)
    return p - y, np.maximum(p * (1 - p), 1e-4)


def logistic_continuation(X, y, s0, c, seed):
    params = dict(objective=logistic_objective, num_leaves=c["leaves"], learning_rate=c["lr"], bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.8, min_data_in_leaf=20, lambda_l2=1.0, max_delta_step=1.0,
                  verbose=-1, num_threads=1, seed=seed)
    return lgb.train(params, lgb.Dataset(X, y, init_score=s0, free_raw_data=False), num_boost_round=c["rounds"])


def run(name, key):
    c = dict(BASE, **CONFIGS[key]); grid = list(range(0, c["rounds"] + 1, max(c["rounds"] // 6, 1)))
    d, feats = load_variant(name, c); A = attributes(name); rho0, rhog = RHO[name]
    fn = OUT + f"preds_{name}_s3_{key}.csv.gz"; frames, done = [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); frames, done = [old], set(old.fold)
    tr_fn = OUT + f"s3_train_{name}_{key}.csv"
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold in done: continue
        t00 = time.time(); seed = zlib.crc32(f"dfm2{name}{fold}".encode()) % 100000
        fit_idx, val_idx = fit_val_split(name, d, tr, seed)
        base_rows = fit_idx if c.get("strict") else tr
        df_ = winsorise_in_window(d, feats, base_rows) if c.get("strict") and name != "IR" else d
        X, y = df_.loc[base_rows, feats].values, df_.loc[base_rows, "y"].values
        Xv, Xt = df_.loc[val_idx, feats].values, df_.loc[te, feats].values
        if c.get("strict"):
            s0, s_out = calibrated_base_grouped(X, y, np.vstack([Xv, Xt]), d.loc[base_rows, "firm"].values, seed=seed)
        else:
            s0, s_out = calibrated_base(X, y, np.vstack([Xv, Xt]), seed=seed)
        s0v, s0t = s_out[:len(val_idx)], s_out[len(val_idx):]
        if not c.get("strict"):                                   # validation rows are training rows: use their oof score
            s0v = s0[pd.Series(np.arange(len(tr)), index=tr).loc[val_idx].values]
        pos = pd.Series(np.arange(len(base_rows)), index=base_rows); vpos = pd.Series(np.arange(len(val_idx)), index=val_idx)
        Xval_full = Xv; base = y.mean()
        rng = np.random.default_rng(seed)
        def mk(idx, k):
            return M2Pool(make_pool_instance(A.loc[idx], rng), d.loc[idx, "y"].values.astype(float), rho0, rhog,
                          seed=seed + k)
        pools, rows = [], []
        for k, idx in enumerate(pools_from(name, d, A, fit_idx, c["ntr"], seed)):
            P = mk(idx, k); r = pos.loc[idx].values
            P.add(M2Solver(P.ins, None, lam=0.0).solve_theta(P.th_real))
            P.solve(sigmoid(s0[r])); P.solve(np.full(len(r), base)); P.solve(np.clip(2 * sigmoid(s0[r]), 0, 1))
            pools.append(P); rows.append(r)
        vpools = [(mk(idx, 5000 + k), vpos.loc[idx].values)
                  for k, idx in enumerate(pools_from(name, d, A, val_idx, c["nval"], seed + 1))]
        t_setup = time.time() - t00
        print(name, key, fold, "setup", f"{t_setup:.0f}s", len(pools), "train pools", len(vpools), "val pools", flush=True)

        def select(predict_raw):
            val = {}
            for R in grid:
                sv = s0v + (predict_raw(Xval_full, R) if R else 0)
                val[R] = float(np.mean([P.regret_exact(sigmoid(sv[r])) for P, r in vpools]))
            return min(val, key=val.get), val

        out, logs = {}, []
        t0 = time.time(); P_ = [copy.copy(P) for P in pools]
        for P, Q in zip(P_, pools): P.cache, P.L = list(Q.cache), list(Q.L)
        b, log = train_correction(X, y, s0, P_, rows, rounds=c["rounds"], lr=c["lr"], num_leaves=c["leaves"],
                                  solve_frac=0.01, seed=seed, eps=c["eps"], M=c["M"], mu=c["mu"])
        pr = lambda Xm, R: b.predict(Xm, num_iteration=R, raw_score=True)
        best, val = select(pr)
        out["DFM2"] = sigmoid(s0t + (pr(Xt, best) if best else 0))
        out["DFM2-max"] = sigmoid(s0t + b.predict(Xt, raw_score=True))
        logs.append(dict(model="DFM2", rounds=best, val=json.dumps({k: round(v * 100, 4) for k, v in val.items()}),
                         cache_mean=float(np.mean([len(P.cache) for P in P_])), seconds=round(time.time() - t0),
                         setup_seconds=round(t_setup), n_train_pools=len(pools)))
        print(name, key, fold, "DFM2", f"{time.time() - t0:.0f}s", "rounds", best, flush=True)
        if key in WITH_CONTROLS:
            out["CAL-BASE"] = sigmoid(s0t)
            t0 = time.time(); bc = logistic_continuation(X, y, s0, c, seed)
            prc = lambda Xm, R: bc.predict(Xm, num_iteration=R, raw_score=True)
            best, val = select(prc)
            out["CE-cont"] = sigmoid(s0t + (prc(Xt, best) if best else 0))
            logs.append(dict(model="CE-cont", rounds=best, val=json.dumps({k: round(v * 100, 4) for k, v in val.items()}),
                             seconds=round(time.time() - t0)))
        for m, p in out.items():
            frames.append(pd.DataFrame(dict(dataset=name, fold=fold, model=m, idx=te, firm=d.loc[te, "firm"].values,
                                            year=d.loc[te, "year"].values, y=d.loc[te, "y"].values, p_raw=p, p_cal=p,
                                            thr=np.nan)))
        pd.concat(frames).to_csv(fn, index=False)
        lg = pd.DataFrame(logs); lg.insert(0, "fold", fold); lg.insert(0, "config", key); lg.insert(0, "dataset", name)
        lg.to_csv(tr_fn, mode="a", header=not os.path.exists(tr_fn), index=False)
        print(name, key, fold, "fold done", f"{time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
