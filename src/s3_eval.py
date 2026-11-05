"""Robustness experiments, evaluation part.

  python s3_eval.py score <dataset>   scores every configuration of s3_train.py on the test pools of rev_eval.py
                                      (same pools and seeds, so results pair pool by pool with the main run)
                                      -> results/decisions_s3_{name}.csv
  python s3_eval.py env <dataset>     robustness of the trained policies to the decision environment (no retraining):
     misspecified environments: decisions are planned with the M2 model of the paper, the realised cost is computed
     in an environment whose structure differs from the planning model and the oracle knows the true environment.
        sev     severity of a distressed supplier unrelated to leverage (mean U(0.2, 0.8)) and more dispersed (phi = 2)
        cap     true capacity = planned capacity x lognormal(0, 0.4); deliveries above it are lost
        link    distress does not map one to one to shortfall: 25% of distressed suppliers deliver normally and 3% of
                the others suffer a distress-size shortfall
        price   realised unit prices deviate from the planned ones (x (1 + N(0, 0.05))), backup premium 0.2, spot 2.2
        all     the four together
     parametric configurations not covered by sensitivity.py: backup fee eta_h, fixed cost f0, severity phi, backup
     premium pi; and the backup reservation and its use under the correlation levels rho x {0, 1, 2}.
     US only: a stratified sample of test pools that contain at least one distressed supplier.
                                      -> results/env_{name}.csv, results/reserve_{name}.csv
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, os, glob, copy, time, zlib, numpy as np, pandas as pd
from scipy.stats import beta as beta_dist
from common import load
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver, ScenarioBank, severity_given_xi, realised_cost
from run_dfm2 import splits_for, RHO
from s3_train import load_variant, CONFIGS

OUT = ROOT + "/results/"
S, B, NPOOL, N = 100, 200, int(os.environ.get("NPOOL", 20)), 20
ENV_MODELS = (("CAL-BASE", "_dfm2"), ("DFM2", "_dfm2"), ("DFM2-max", "_dfm2"), ("CE-cont", "_rev"),
              ("SPO+-warm", "_rev"), ("PFYL-warm", "_rev"), ("LGBM-Focal", ""), ("LGBM", ""), ("XGB", ""))
RESERVE_MODELS = ("CAL-BASE", "DFM2", "LGBM-Focal")
PARAM = [("eta_h=0.02", {"eta_h": 0.02}), ("eta_h=0.10", {"eta_h": 0.10}), ("f0=0", {"f0": 0.0}),
         ("f0=0.03", {"f0": 0.03}), ("phi=2", {"phi": 2.0}), ("phi=8", {"phi": 8.0}), ("pi=0.05", {"pi": 0.05}),
         ("pi=0.20", {"pi": 0.20})]


def nreg(ins, z, th, oracle):
    return (float(realised_cost(ins, *z, th).mean()) - oracle) / oracle


def oracle_of(ins, th):
    return float(realised_cost(ins, *M2Solver(ins, None, lam=0.0).solve_theta(th), th).mean())


# ---------------------------------------------------------------- scoring of the s3 configurations
def score(name):
    rho0, rhog = RHO[name]; A = attributes(name)
    fn = OUT + f"decisions_s3_{name}.csv"; rows, done = [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); rows = old.to_dict("records")
        done = set(zip(old.config, old.fold))
    for f in sorted(glob.glob(OUT + f"preds_{name}_s3_*.csv.gz")):
        key = f.split("_s3_")[1].split(".")[0]
        d, _ = load_variant(name, dict(CONFIGS[key]))
        P = pd.read_csv(f, usecols=["fold", "model", "idx", "p_cal"]); P["fold"] = P.fold.astype(str)
        for fold, tr, te in splits_for(name, d):
            fold = str(fold)
            if (key, fold) in done or fold not in set(P.fold): continue
            t0 = time.time(); Pf = P[P.fold == fold].set_index(["model", "idx"]); base = float(d.loc[tr, "y"].mean())
            for k, idx in enumerate(draw_pools(te, A, NPOOL, N, seed=zlib.crc32(f"{name}{fold}".encode()))):
                rng = np.random.default_rng([zlib.crc32(f"{name}{fold}".encode()), k])
                ins = make_pool_instance(A.loc[idx], rng)
                xi = d.loc[idx, "y"].values.astype(float); th = severity_given_xi(ins, xi, B, seed=k + 7)
                m2 = M2Solver(ins, ScenarioBank(ins, S, rho0, rhog, seed=k + 11)); om = oracle_of(ins, th)
                res = {"NAIVE": m2.solve(np.full(N, base))}
                for m in Pf.index.get_level_values(0).unique():
                    res[m] = m2.solve(Pf.loc[m].loc[idx, "p_cal"].values)
                for m, z in res.items():
                    rows.append(dict(dataset=name, config=key, fold=fold, pool=k, model=m, nregret=nreg(ins, z, th, om),
                                     n_distress=int(xi.sum())))
            pd.DataFrame(rows).to_csv(fn, index=False)
            print(name, "score", key, fold, f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- misspecified environments
def true_env(ins, xi, kind, rng):
    """Return (ins_true, theta_true, cap_true or None) for one pool."""
    it = copy.copy(ins); n = ins.N; cap = None
    kinds = ("sev", "cap", "link", "price") if kind == "all" else (kind,)
    mu, phi = ins.mu.copy(), ins.phi
    if "sev" in kinds: mu, phi = rng.uniform(0.2, 0.8, n), 2.0
    UB, Ub = rng.random((B, n)), rng.random((B, n))
    Bd = beta_dist.ppf(UB, (mu * phi)[None, :], ((1 - mu) * phi)[None, :])
    bd = beta_dist.ppf(Ub, ins.delta0 * ins.phi0, (1 - ins.delta0) * ins.phi0)
    X = np.repeat(xi[None, :], B, 0)
    if "link" in kinds:
        u = rng.random((B, n)); X = np.where(X == 1, (u >= 0.25).astype(float), (u < 0.03).astype(float))
    th = X * Bd + (1 - X) * bd
    if "cap" in kinds:
        cap = np.clip(ins.K * rng.lognormal(0, 0.4, n), 0.10, 1.0); it.K = cap; it.u = np.minimum(cap, ins.sbar * ins.D)
    if "price" in kinds:
        it.c = ins.c * (1 + rng.normal(0, 0.05, n)); it.pi, it.e = 0.20, 2.2
    return it, th, cap


def cost_true(it, z, th, cap):
    y, q, v = z
    if cap is None: return float(realised_cost(it, y, q, v, th).mean())
    qe = np.minimum(q, cap); ve = np.clip(np.minimum(v, cap - qe), 0, None)
    lost = np.where(q > 0, 1 - qe / np.maximum(q, 1e-12), 0.0)
    th_e = 1 - (1 - th) * (1 - lost)[None, :]
    return float(realised_cost(it, y, q, ve, th_e).mean())


def backup_stats(ins, z, th):
    y, q, v = z
    short = (th * q[None, :]).sum(1); avail = ((1 - th) * v[None, :]).sum(1)
    return float(v.sum()), float(np.minimum(short, avail).mean()), float(y.sum())


def env(name):
    d, _ = load(name); A = attributes(name); rho0, rhog = RHO[name]
    P = {}
    for _, sfx in ENV_MODELS:
        if sfx not in P:
            t = pd.read_csv(OUT + f"preds_{name}{sfx}.csv.gz", usecols=["fold", "model", "idx", "p_cal"])
            t["fold"] = t.fold.astype(str); P[sfx] = t
    fn, fr = OUT + f"env_{name}.csv", OUT + f"reserve_{name}.csv"
    rows, rrows, done = [], [], set()
    if os.path.exists(fn):
        old = pd.read_csv(fn); old["fold"] = old.fold.astype(str); rows = old.to_dict("records"); done = set(old.fold)
        if os.path.exists(fr): rrows = pd.read_csv(fr).to_dict("records")
    for fold, tr, te in splits_for(name, d):
        fold = str(fold)
        if fold in done: continue
        t0 = time.time(); base = float(d.loc[tr, "y"].mean()); key = zlib.crc32(f"{name}{fold}".encode())
        probs = {m: P[s][(P[s].fold == fold) & (P[s].model == m)].set_index("idx").p_cal for m, s in ENV_MODELS}
        samples = [("all", draw_pools(te, A, NPOOL, N, seed=key))]
        if name == "US":
            samples.append(("stratified", stratified_pools(te, d, A, NPOOL, seed=zlib.crc32(f"strat{name}{fold}".encode()))))
        for tag, pools in samples:
            for k, idx in enumerate(pools):
                rng = np.random.default_rng([key if tag == "all" else key + 1, k])
                ins = make_pool_instance(A.loc[idx], rng)
                xi = d.loc[idx, "y"].values.astype(float); th = severity_given_xi(ins, xi, B, seed=k + 7)
                m2 = M2Solver(ins, ScenarioBank(ins, S, rho0, rhog, seed=k + 11)); om = oracle_of(ins, th)
                dec = {"NAIVE": m2.solve(np.full(N, base))}
                for m in probs: dec[m] = m2.solve(probs[m].loc[idx].values)
                for m, z in dec.items():
                    rows.append(dict(dataset=name, fold=fold, sample=tag, pool=k, env="base", model=m,
                                     nregret=nreg(ins, z, th, om), n_distress=int(xi.sum())))
                if tag != "all": continue
                erng = np.random.default_rng([key, k, 99])
                for kind in ("sev", "cap", "link", "price", "all"):
                    it, tht, cap = true_env(ins, xi, kind, erng); o = oracle_of(it, tht)
                    for m, z in dec.items():
                        rows.append(dict(dataset=name, fold=fold, sample=tag, pool=k, env=kind, model=m,
                                         nregret=(cost_true(it, z, tht, cap) - o) / o, n_distress=int(xi.sum())))
                if k < 10:                                         # parametric configurations: first 10 pools
                    for cname, kw in PARAM:
                        ins_c = make_pool_instance(A.loc[idx], np.random.default_rng([key, k]), **kw)
                        th_c = severity_given_xi(ins_c, xi, B, seed=k + 7); o = oracle_of(ins_c, th_c)
                        mc = M2Solver(ins_c, ScenarioBank(ins_c, S, rho0, rhog, seed=k + 11))
                        for m in ("NAIVE",) + tuple(probs):
                            p = np.full(N, base) if m == "NAIVE" else probs[m].loc[idx].values
                            rows.append(dict(dataset=name, fold=fold, sample=tag, pool=k, env=cname, model=m,
                                             nregret=nreg(ins_c, mc.solve(p), th_c, o), n_distress=int(xi.sum())))
                for mul in (0.0, 1.0, 2.0):
                    mr = M2Solver(ins, ScenarioBank(ins, S, min(rho0 * mul, 0.95), min(rhog * mul, 0.95), seed=k + 11))
                    for m in RESERVE_MODELS:
                        z = mr.solve(probs[m].loc[idx].values); vr, used, ns = backup_stats(ins, z, th)
                        rrows.append(dict(dataset=name, fold=fold, pool=k, rho_mult=mul, model=m, reserved=vr,
                                          backup_used=used, n_selected=ns, nregret=nreg(ins, z, th, om)))
        pd.DataFrame(rows).to_csv(fn, index=False); pd.DataFrame(rrows).to_csv(fr, index=False)
        print(name, "env", fold, f"{time.time() - t0:.0f}s", flush=True)


def stratified_pools(te, d, A, n_pools, seed, min_groups=3):
    """Pools of size N with at least one distressed supplier (one drawn from the distressed firms, the rest at random)."""
    rng = np.random.default_rng(seed)
    elig = np.array([i for i in te if A.pool.get(i, 1) == 1]); posi = elig[d.loc[elig, "y"].values == 1]
    if len(posi) == 0 or len(elig) < N: return []
    out = []
    for _ in range(n_pools * 20):
        first = rng.choice(posi); rest = rng.choice(elig[elig != first], N - 1, replace=False)
        sel = np.sort(np.r_[first, rest]); gs = A.loc[sel, "group"]
        if gs.isna().all() or gs.nunique() >= min_groups: out.append(sel)
        if len(out) == n_pools: break
    return out


if __name__ == "__main__":
    {"score": score, "env": env}[sys.argv[1]](sys.argv[2])
