"""Step 6 — DF-Distress and decision-focused benchmarks on one gradient-boosting backbone (LightGBM custom objective).

Score s_r (log-odds) per firm-year r, p_r = sigmoid(s_r). Training pools k = sets of N firm-years of one decision year.
M1 cost vector of pool k:  w_q = a + beta * p   (affine, model_formulation_v3_final §6).
Oracle decision z^o_k = z*_M1(w(xi_k)) with realised labels xi.

Gradients w.r.t. the q-part of w, chained to scores by dw/ds = beta * p(1-p):
  dfd  (ours): PFYL  g = z^o - E[z*(w+eps Z)];  curvature = Stein diag -(1/eps) E[(z*(w+eps Z)-z*(w))_j Z_j] >= 0
               (Gauss-Newton);  + calibration mu_m (p - y), mu_m annealed to a floor mu_inf;  weights omega ∝ beta u
  pfyl       : PFYL gradient, constant curvature, no calibration, no weights          (Berthet et al. 2020)
  spo        : SPO+ subgradient 2(z^o - z*(2w - w_true)), constant curvature           (Elmachtoub & Grigas 2022)
  dbb        : blackbox differentiation of regret, (z*(w+lam w_true) - z*(w))/lam      (Vlastelica et al. 2020)
Solution caching (Mulamba et al. 2021): a fraction `solve_frac` of pool-level oracle calls is solved exactly; the rest
take the best cached solution of that pool.
Safeguards common to all methods: decision gradients rescaled once to the log-loss gradient scale, Hessian floor,
LightGBM max_delta_step and L2 leaf penalty.
"""
import numpy as np, lightgbm as lgb
from ssoa import M1Solver

def sigmoid(s): return 1 / (1 + np.exp(-np.clip(s, -30, 30)))

class PoolSet:
    """Training pools: row positions (into the training matrix) and their Instances."""
    def __init__(self, pools, instances, y):
        self.rows, self.ins, self.y = pools, instances, np.asarray(y, float)
        self.solvers = [M1Solver(ins) for ins in instances]
        self.w_true = [s.cost_vector(self.y[r]) for s, r in zip(self.solvers, pools)]
        self.zo = [s.solve_w(w) for s, w in zip(self.solvers, self.w_true)]
        for s in self.solvers:                              # seed caches: risk-free and risk-blind decisions
            s.solve(np.zeros(s.ins.N)); s.solve(np.full(s.ins.N, 0.1))
    def oracle(self, k, W, rng, solve_frac):
        s = self.solvers[k]
        if solve_frac >= 1 or rng.random() < solve_frac:
            return np.array([s.solve_w(w) for w in W])
        return s.best_in_cache(W)
    def regret(self, p, solve_frac=1.0, rng=None):
        """Mean normalised M1 regret of plug-in decisions under probabilities p (per-row array)."""
        rng = rng or np.random.default_rng(0); out = []
        for k, rows in enumerate(self.rows):
            z = self.oracle(k, self.solvers[k].cost_vector(p[rows])[None, :], rng, solve_frac)[0]
            wt, zo = self.w_true[k], self.zo[k]; out.append((wt @ z - wt @ zo) / (wt @ zo))
        return float(np.mean(out))

def make_objective(method, P, n_rows, *, eps_rel=0.5, M=8, solve_frac=0.1, mu0=1.0, mu_inf=0.1, decay=0.97,
                   lam_dbb=1.0, seed=0, weights=True, stein=True, calib=True, log=None):
    rng = np.random.default_rng(seed); state = {"it": 0, "scale": None}
    def fobj(preds, data):
        y = data.get_label(); p = sigmoid(preds)
        if method == "ce":                                  # control: same backbone, log-loss only
            state["it"] += 1; return p - y, np.maximum(p * (1 - p), 1e-4)
        G = np.zeros(n_rows); H = np.zeros(n_rows); cnt = np.zeros(n_rows); reg = []
        for k, rows in enumerate(P.rows):
            ins, sol = P.ins[k], P.solvers[k]; N = ins.N
            pk = p[rows]; w = sol.cost_vector(pk); wt = P.w_true[k]; zo = P.zo[k]
            dw_ds = ins.beta * pk * (1 - pk)
            om = ins.beta * ins.u if (method == "dfd" and weights) else np.ones(N); om = om / om.mean()
            if method in ("dfd", "pfyl"):
                eps = eps_rel * max(np.std(w[N:]), 1e-3)
                Z = rng.standard_normal((M, N)); Wp = np.repeat(w[None, :], M, 0); Wp[:, N:] += eps * Z
                Zs = P.oracle(k, np.vstack([w[None, :], Wp]), rng, solve_frac)
                zhat, zp = Zs[0], Zs[1:, N:]
                gq = zo[N:] - zp.mean(0)
                hq = (np.maximum(-((zp - zhat[None, N:]) * Z).mean(0) / eps, 0) + 1e-2 * ins.D
                      if (method == "dfd" and stein) else None)
            elif method == "spo":
                Zs = P.oracle(k, np.vstack([w, 2 * w - wt]), rng, solve_frac)
                zhat = Zs[0]; gq = 2 * (zo[N:] - Zs[1][N:]); hq = None
            elif method == "dbb":
                Zs = P.oracle(k, np.vstack([w, w + lam_dbb * wt]), rng, solve_frac)
                zhat = Zs[0]; gq = (Zs[1][N:] - zhat[N:]) / lam_dbb; hq = None
            else:
                raise ValueError(method)
            reg.append((wt @ zhat - wt @ zo) / (wt @ zo))
            G[rows] += om * gq * dw_ds
            H[rows] += (om * hq * dw_ds ** 2) if hq is not None else 0.0
            cnt[rows] += 1
        m = np.maximum(cnt, 1); G /= m; H /= m
        if state["scale"] is None:                          # one-off rescaling to the log-loss gradient scale
            state["scale"] = np.abs(p - y).mean() / max(np.abs(G[cnt > 0]).mean(), 1e-12)
        G *= state["scale"]
        if method == "dfd" and stein:
            H *= state["scale"]
        else:
            H = np.where(cnt > 0, np.mean(p * (1 - p)), 0.0)   # constant curvature on pooled rows
        if method == "dfd" and calib:
            mu = mu_inf + (mu0 - mu_inf) * decay ** state["it"]
            G = G + mu * (p - y); H = H + mu * p * (1 - p)
        H = np.maximum(H, 1e-4)
        if log is not None: log.append(dict(it=state["it"], regret=float(np.mean(reg))))
        state["it"] += 1
        return G, H
    return fobj

def train_df(method, X, y, P, *, init_score=None, rounds=200, lr=0.05, num_leaves=15, seed=0, **kw):
    log = []
    fobj = make_objective(method, P, len(y), seed=seed, log=log, **kw)
    ds = lgb.Dataset(X, y, init_score=init_score, free_raw_data=False)
    params = dict(objective=fobj, num_leaves=num_leaves, learning_rate=lr, bagging_fraction=0.8, bagging_freq=1,
                  feature_fraction=0.8, min_data_in_leaf=10, lambda_l2=1.0, max_delta_step=1.0,
                  verbose=-1, num_threads=1, seed=seed)
    b = lgb.train(params, ds, num_boost_round=rounds)
    return b, log

def predict_df(b, X, init_score=None):
    s = b.predict(X, raw_score=True)
    return sigmoid(s + (0 if init_score is None else init_score))
