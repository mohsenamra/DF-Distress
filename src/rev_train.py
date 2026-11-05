"""Comparator experiments, training part. One run per dataset; resumable by fold.

Variants trained per fold (same seeds, base layer and training pools as run_dfm2.py):
  DFM2-MC        correction trained on a mean-CVaR decision loss across training pools: each pool's normalised regret
                 is weighted by (1 - lambda) + lambda / (1 - alpha) if it lies in the worst (1 - alpha) share of pools,
                 by (1 - lambda) otherwise (lambda = 0.3, alpha = 0.9), the empirical counterpart of the M2 criterion
                 over outcome realisations; rounds selected by the same criterion on the validation pools
  DFM2-noOracle  DFM2 with the ex-post optimal decision removed from the initial cache
  CE-cont        matched control: the calibrated base continued with the logistic loss only
                 (same leaves, learning rate, rounds and validation selection as DFM2)
  SPO+-warm      SPO+ through M1, started from the calibrated base, same capacity, budget and selection
  PFYL-warm      perturbed Fenchel-Young loss through M1, started from the calibrated base, same capacity, budget, selection
Outputs:
  results/preds_{name}_rev.csv.gz   out-of-sample probabilities (same format as the other prediction files)
  results/rev_cache_{name}.csv      cache diagnostics: every exact solve during training compared with the cached choice
  results/rev_grad_{name}.csv       cosine similarity between cached and exact-solve gradient estimates
  results/rev_train_{name}.csv      validation curves, selected rounds, cache sizes and timings
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, os, time, zlib, json, numpy as np, pandas as pd, lightgbm as lgb
from common import load
from instances import attributes, make_pool_instance
from ssoa import M2Solver, realised_cost
from dfm2 import calibrated_base, M2Pool, realised_cost_batch, mean_cvar_rows, sigmoid, logit
from dfd import PoolSet, train_df
from run_dfm2 import splits_for, fit_val_split, pools_from, RHO

OUT = ROOT + "/results/"
NTR, NVAL, RND, LR, LEAVES = 60, 15, 150, 0.08, 4
EPS, M, MU, SOLVE_FRAC = 0.5, 8, 0.3, 0.01
LAM, ALPHA = 0.3, 0.9
if os.environ.get("REV_SMOKE"): NTR, NVAL, RND = 3, 2, 6
CHECK = list(range(0, RND + 1, max(RND // 6, 1)))


def mc_cost(costs, lam=LAM, alpha=ALPHA):
    """(1 - lam) mean + lam CVaR_alpha of equally likely costs (exact for a finite sample)."""
    return float(mean_cvar_rows(np.asarray(costs, float)[None, :], lam, alpha)[0])


class RevPool(M2Pool):
    """M2 training/validation pool that also keeps the ex-post optimal decision so it can be added to a cache or not."""
    def __init__(self, ins, xi_real, rho0, rhog, seed, S=100, B=50):
        super().__init__(ins, xi_real, rho0, rhog, seed, S, B)
        self.z_or = M2Solver(ins, None, lam=0.0).solve_theta(self.th_real)
    def saa_obj(self, p, Z):
        th = self.bank.draw(p)[1]
        return mean_cvar_rows(realised_cost_batch(self.ins, Z, th), LAM, ALPHA)
    def regret_exact(self, p):
        z = self.solver.solve(p)
        return (float(realised_cost(self.ins, *z, self.th_real).mean()) - self.oracle_cost) / self.oracle_cost
    def clone(self, with_oracle):
        Q = RevPool.__new__(RevPool); Q.__dict__.update(self.__dict__)
        Q.cache, Q.L = list(self.cache), list(self.L)
        if with_oracle: Q.add(self.z_or)
        return Q


def tail_weights(reg, lam=LAM, alpha=ALPHA):
    """Subgradient weights of (1 - lam) mean + lam CVaR_alpha of the pool regrets, scaled to mean one."""
    K = len(reg); k = max(int(np.ceil((1 - alpha) * K)), 1)
    w = np.full(K, 1 - lam); w[np.argsort(reg)[-k:]] += lam * K / k
    return w


def pooled_mc(reg, lam=LAM, alpha=ALPHA):
    reg = np.sort(np.asarray(reg, float)); k = max(int(np.ceil((1 - alpha) * len(reg))), 1)
    return float((1 - lam) * reg.mean() + lam * reg[-k:].mean())


def make_objective(pools, rows, n_rows, *, loss="mean", diag=None, seed=0, log=None):
    """Score-function gradient (paired antithetic form) of the cached realised cost plus the logistic calibration term.
    loss = "mean": every pool weighted equally; loss = "tail": mean-CVaR weights across pools (tail_weights)."""
    rng = np.random.default_rng(seed); st = {"scale": None, "it": 0}
    def fobj(preds, data):
        y = data.get_label(); p = sigmoid(preds)
        G = np.zeros(n_rows); cnt = np.zeros(n_rows); per = []
        for k, (P, r) in enumerate(zip(pools, rows)):
            s = preds[r]; N = len(r)
            Zh = rng.standard_normal((M // 2, N)); Z = np.vstack([Zh, -Zh])
            L = np.empty(M)
            for m in range(M):
                pm = sigmoid(s + EPS * Z[m])
                if rng.random() < SOLVE_FRAC:
                    kc = P.choose(pm); zc = P.cache[kc]
                    ze = P.solver.solve(pm)
                    if diag is not None:
                        o = P.saa_obj(pm, [zc, ze])
                        le = float(realised_cost(P.ins, *ze, P.th_real).mean())
                        same = bool(np.allclose(zc[0], ze[0]) and np.allclose(zc[1], ze[1], atol=1e-5)
                                    and np.allclose(zc[2], ze[2], atol=1e-5))
                        diag.append(dict(it=st["it"], pool=k, cache_size=len(P.cache), same_decision=same,
                                         saa_gap_rel=float((o[0] - o[1]) / abs(o[1])),
                                         loss_gap_rel=float((P.L[kc] - le) / P.oracle_cost)))
                    P.add(ze)
                L[m] = P.L[P.choose(pm)]
            L = L / P.oracle_cost
            h = M // 2
            per.append((r, ((L[:h] - L[h:])[:, None] * Zh).sum(0) / (M * EPS), L.mean() - 1))
        reg = np.array([x[2] for x in per])
        w = tail_weights(reg) if loss == "tail" else np.ones(len(per))
        for (r, g, _), wk in zip(per, w):
            G[r] += wk * g; cnt[r] += 1
        G /= np.maximum(cnt, 1)
        if st["scale"] is None:
            st["scale"] = np.abs(p - y).mean() / max(np.abs(G[cnt > 0]).mean(), 1e-12)
        G = st["scale"] * G + MU * (p - y)
        H = np.where(cnt > 0, np.mean(p * (1 - p)), 0.0) + MU * p * (1 - p)
        if log is not None: log.append(float(pooled_mc(reg) if loss == "tail" else reg.mean()))
        st["it"] += 1
        return G, np.maximum(H, 1e-4)
    return fobj


def logistic_objective(preds, data):
    y = data.get_label(); p = sigmoid(preds)
    return p - y, np.maximum(p * (1 - p), 1e-4)


def boost(X, y, init, fobj, seed):
    ds = lgb.Dataset(X, y, init_score=init, free_raw_data=False)
    params = dict(objective=fobj, num_leaves=LEAVES, learning_rate=LR, bagging_fraction=0.8, bagging_freq=1,
                  feature_fraction=0.8, min_data_in_leaf=20, lambda_l2=1.0, max_delta_step=1.0, verbose=-1,
                  num_threads=1, seed=seed)
    return lgb.train(params, ds, num_boost_round=RND)


def gradient_check(pools, rows, scores, rng, n_pools=5):
    """Cosine similarity between the cached and the exact-solve score-function gradient on the same perturbations."""
    out = []
    for k in rng.choice(len(pools), min(n_pools, len(pools)), replace=False):
        P, r = pools[k], rows[k]; s = scores[r]; N = len(r)
        Zh = rng.standard_normal((M // 2, N)); Z = np.vstack([Zh, -Zh])
        Lc, Le = np.empty(M), np.empty(M)
        for m in range(M):
            pm = sigmoid(s + EPS * Z[m])
            kc = P.choose(pm); Lc[m] = P.L[kc]
            ce = realised_cost(P.ins, *P.solver.solve(pm), P.th_real)
            Le[m] = float(ce.mean())
        h = M // 2
        gc = ((Lc[:h] - Lc[h:])[:, None] * Zh).sum(0); ge = ((Le[:h] - Le[h:])[:, None] * Zh).sum(0)
        nc, ne = np.linalg.norm(gc), np.linalg.norm(ge)
        out.append(dict(pool=int(k), cosine=float(gc @ ge / (nc * ne)) if nc > 0 and ne > 0 else np.nan,
                        loss_gap_rel=float(np.mean(np.abs(Lc - Le)) / P.oracle_cost)))
    return out


def run(name):
    d, feats = load(name); A = attributes(name); rho0, rhog = RHO[name]
    fn = OUT + f"preds_{name}_rev.csv.gz"; frames, done = [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); frames, done = [old], set(old.fold)
    diag_fn, grad_fn, tr_fn = OUT + f"rev_cache_{name}.csv", OUT + f"rev_grad_{name}.csv", OUT + f"rev_train_{name}.csv"
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
            return RevPool(make_pool_instance(A.loc[idx], rng), d.loc[idx, "y"].values.astype(float), rho0, rhog,
                           seed=seed + k)
        pools, rows = [], []
        for k, idx in enumerate(pools_from(name, d, A, fit_idx, NTR, seed)):
            P = mk(idx, k); r = pos.loc[idx].values
            P.solve(sigmoid(s0[r])); P.solve(np.full(len(r), base)); P.solve(np.minimum(2 * sigmoid(s0[r]), 0.999))
            pools.append(P); rows.append(r)
        vpools = [(mk(idx, 5000 + k), pos.loc[idx].values)
                  for k, idx in enumerate(pools_from(name, d, A, val_idx, NVAL, seed + 1))]
        t_setup = time.time() - t00
        print(name, fold, "setup", f"{t_setup:.0f}s", len(pools), "train pools", len(vpools), "val pools", flush=True)
        out, diags, grads, trains = {}, [], [], []

        def select(predict_raw, init, initt, mc):
            val = {}
            for R in CHECK:
                sv = init + (predict_raw(X, R) if R else 0)
                rg = [P.regret_exact(sigmoid(sv[r])) for P, r in vpools]
                val[R] = pooled_mc(rg) if mc else float(np.mean(rg))
            best = min(val, key=val.get)
            return best, val

        for label, loss, with_or in (("DFM2-MC", "tail", True), ("DFM2-noOracle", "mean", False)):
            t0 = time.time(); P_ = [P.clone(with_or) for P in pools]
            dg, log = [], []
            b = boost(X, y, s0, make_objective(P_, rows, len(y), loss=loss, diag=dg, seed=seed, log=log), seed)
            pr = lambda Xm, R: b.predict(Xm, num_iteration=R, raw_score=True)
            best, val = select(pr, s0, s0t, loss == "tail")
            out[label] = sigmoid(s0t + (pr(Xt, best) if best else 0))
            if label == "DFM2-MC": out["DFM2-MC-max"] = sigmoid(s0t + b.predict(Xt, raw_score=True))
            grng = np.random.default_rng(seed + 77)
            for g in gradient_check(P_, rows, s0, grng): grads.append(dict(g, stage="start", variant=label))
            for g in gradient_check(P_, rows, s0 + b.predict(X, raw_score=True), grng):
                grads.append(dict(g, stage="end", variant=label))
            for x in dg: diags.append(dict(x, variant=label))
            sizes = [len(P.cache) for P in P_]
            trains.append(dict(variant=label, rounds=best, val=json.dumps({k: round(v * 100, 4) for k, v in val.items()}),
                               cache_mean=float(np.mean(sizes)), cache_max=int(np.max(sizes)), exact_solves=len(dg),
                               seconds=round(time.time() - t0), setup_seconds=round(t_setup)))
            print(name, fold, label, f"{time.time() - t0:.0f}s", "rounds", best, "cache", round(np.mean(sizes), 1),
                  "exact solves", len(dg), flush=True)

        t0 = time.time()
        b = boost(X, y, s0, logistic_objective, seed)
        pr = lambda Xm, R: b.predict(Xm, num_iteration=R, raw_score=True)
        best, val = select(pr, s0, s0t, False)
        out["CE-cont"] = sigmoid(s0t + (pr(Xt, best) if best else 0))
        trains.append(dict(variant="CE-cont", rounds=best, val=json.dumps({k: round(v * 100, 4) for k, v in val.items()}),
                           seconds=round(time.time() - t0)))
        print(name, fold, "CE-cont", f"{time.time() - t0:.0f}s", "rounds", best, flush=True)

        PS = PoolSet(rows, [P.ins for P in pools], y)
        for label, meth in (("SPO+-warm", "spo"), ("PFYL-warm", "pfyl")):
            t0 = time.time()
            b, _ = train_df(meth, X, y, PS, init_score=s0, rounds=RND, lr=LR, num_leaves=LEAVES, seed=seed)
            pr = lambda Xm, R, b=b: b.predict(Xm, num_iteration=R, raw_score=True)
            best, val = select(pr, s0, s0t, False)
            out[label] = sigmoid(s0t + (pr(Xt, best) if best else 0))
            trains.append(dict(variant=label, rounds=best, val=json.dumps({k: round(v * 100, 4) for k, v in val.items()}),
                               seconds=round(time.time() - t0)))
            print(name, fold, label, f"{time.time() - t0:.0f}s", "rounds", best, flush=True)

        for m, p in out.items():
            frames.append(pd.DataFrame(dict(dataset=name, fold=fold, model=m, idx=te, firm=d.loc[te, "firm"].values,
                                            year=d.loc[te, "year"].values, y=d.loc[te, "y"].values, p_raw=p, p_cal=p,
                                            thr=np.nan)))
        pd.concat(frames).to_csv(fn, index=False)
        for rows_, f in ((diags, diag_fn), (grads, grad_fn), (trains, tr_fn)):
            if rows_:
                df_ = pd.DataFrame(rows_); df_.insert(0, "fold", fold); df_.insert(0, "dataset", name)
                df_.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        print(name, fold, "fold done", f"{time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]: run(n)
