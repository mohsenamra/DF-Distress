"""DF-Distress-M2: decision-focused correction trained on the full two-stage model M2 (model_formulation_v3 §5–7).

  1. Base layer (cross-fitted, calibrated):  s0 = logit Platt(LGBM)  — out-of-fold on training rows, so the
     decision signal is not killed by in-sample overfitting; the same Platt map is applied to the full-train model at test.
  2. Decision-focused correction: boosted trees f_m added to s0, trained with
        G = kappa * g_DF + mu (p - y),   g_DF = score-function (perturbed-optimiser) gradient of the REALISED M2 cost
        g_DF,i = 1/(M eps) sum_m (L(z*_M2(sigmoid(s + eps Z_m))) - L_bar) Z_m,i      (antithetic Z)
     where L(z) = Lemma-1 realised cost of first-stage decision z under the pool's realised distress (severity averaged).
     z*_M2 is evaluated over a per-pool cache of exact M2 solutions (solution caching, Mulamba et al. 2021): the cached
     decision with the lowest SAA mean-CVaR under scenarios drawn from p (common random numbers); a fraction of calls
     solves M2 exactly and grows the cache.
  3. Number of correction rounds chosen on held-out validation pools (last training year / 20% hold-out) by exact-M2
     regret, from {0, 15, 30, 45, 60}; 0 = calibrated base model (so validation never prefers a worse model).
"""
import numpy as np, lightgbm as lgb
from scipy.stats import norm, beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ssoa import M2Solver, severity_given_xi, realised_cost, mean_cvar

def sigmoid(s): return 1 / (1 + np.exp(-np.clip(s, -30, 30)))
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

LGB_BASE = dict(objective="binary", num_leaves=15, learning_rate=0.05, bagging_fraction=0.8, bagging_freq=1,
                feature_fraction=0.8, min_data_in_leaf=10, verbose=-1, num_threads=1)

def calibrated_base(X, y, Xte, seed=0, k=5):
    """Out-of-fold calibrated logits for training rows and Platt-mapped full-model logits for test rows."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        b = lgb.train(dict(LGB_BASE, seed=seed), lgb.Dataset(X[tr], y[tr]), 300)
        oof[va] = b.predict(X[va], raw_score=True)
    pl = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)
    full = lgb.train(dict(LGB_BASE, seed=seed), lgb.Dataset(X, y), 300).predict(Xte, raw_score=True)
    a, c = pl.coef_[0, 0], pl.intercept_[0]
    return a * oof + c, a * full + c

class FastBank:
    """ScenarioBank with severities precomputed (they do not depend on p); only distress indicators depend on p."""
    def __init__(self, ins, S, rho0, rhog, seed=0):
        rng = np.random.default_rng(seed); N = ins.N
        groups, gi = np.unique(ins.group, return_inverse=True)
        M0 = rng.standard_normal(S); Mg = rng.standard_normal((S, len(groups))); eps = rng.standard_normal((S, N))
        self.A = np.sqrt(rho0) * M0[:, None] + np.sqrt(max(rhog - rho0, 0)) * Mg[:, gi] + np.sqrt(1 - rhog) * eps
        UB = rng.random((S, N)); Ub = rng.random((S, N))
        self.B = beta_dist.ppf(UB, (ins.mu * ins.phi)[None, :], ((1 - ins.mu) * ins.phi)[None, :])
        self.b = beta_dist.ppf(Ub, ins.delta0 * ins.phi0, (1 - ins.delta0) * ins.phi0)
        self.S, self.ins = S, ins
    def draw(self, p):
        xi = (self.A <= norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))[None, :]).astype(float)
        return xi, xi * self.B + (1 - xi) * self.b

def realised_cost_batch(ins, cache, theta):
    """Lemma-1 realised cost of K first-stage decisions under S scenarios at once; returns (K, S)."""
    Y = np.array([z[0] for z in cache]); Q = np.array([z[1] for z in cache]); V = np.array([z[2] for z in cache])
    first = Y @ ins.f + V @ ins.h                                         # (K,)
    prim = Q @ (ins.c[None, :] * (1 - theta)).T                           # (K, S)
    short = Q @ theta.T                                                   # (K, S)
    order = np.argsort(ins.c); price = (1 + ins.pi) * ins.c[order]
    avail = (1 - theta)[None, :, order] * V[:, None, order]               # (K, S, N)
    cum = np.cumsum(avail, axis=2)
    before = cum - avail
    use = np.clip(short[:, :, None] - before, 0, avail)
    rec = (use * price).sum(2) + ins.e * np.clip(short - cum[:, :, -1], 0, None)
    return first[:, None] + prim + rec

def mean_cvar_rows(C, lam=0.3, alpha=0.9):
    S = C.shape[1]; k = int(np.ceil(alpha * S)); eta = np.sort(C, axis=1)[:, k - 1]
    return (1 - lam) * C.mean(1) + lam * (eta + np.maximum(C - eta[:, None], 0).mean(1) / (1 - alpha))

class M2Pool:
    def __init__(self, ins, xi_real, rho0, rhog, seed, S=100, B=50):
        self.ins, self.bank = ins, FastBank(ins, S, rho0, rhog, seed)
        self.th_real = severity_given_xi(ins, xi_real, B, seed=seed + 3)
        self.solver = M2Solver(ins, self.bank)
        self.cache, self.L = [], []
        self.oracle_cost = self._real(M2Solver(ins, None, lam=0.0).solve_theta(self.th_real))
    def _real(self, z): return float(realised_cost(self.ins, *z, self.th_real).mean())
    def add(self, z):
        for c in self.cache:
            if np.allclose(c[0], z[0]) and np.allclose(c[1], z[1], atol=1e-5) and np.allclose(c[2], z[2], atol=1e-5):
                return
        self.cache.append(z); self.L.append(self._real(z))
    def solve(self, p):
        z = self.solver.solve(p); self.add(z); return z
    def choose(self, p):
        th = self.bank.draw(p)[1]
        C = realised_cost_batch(self.ins, self.cache, th)               # (K, S) costs of all cached decisions
        return int(np.argmin(mean_cvar_rows(C)))
    def regret_exact(self, p):
        return (self._real(self.solver.solve(p)) - self.oracle_cost) / self.oracle_cost

def make_objective(pools, rows, n_rows, *, eps=0.5, M=8, solve_frac=0.03, mu=0.3, seed=0, log=None):
    rng = np.random.default_rng(seed); st = {"scale": None, "it": 0}
    def fobj(preds, data):
        y = data.get_label(); p = sigmoid(preds)
        G = np.zeros(n_rows); cnt = np.zeros(n_rows); reg = []
        for P, r in zip(pools, rows):
            s = preds[r]; N = len(r)
            Z = rng.standard_normal((M // 2, N)); Z = np.vstack([Z, -Z])
            L = np.empty(M)
            for m in range(M):
                pm = sigmoid(s + eps * Z[m])
                if rng.random() < solve_frac: P.solve(pm)
                L[m] = P.L[P.choose(pm)]
            G[r] += ((L - L.mean())[:, None] * Z).mean(0) / eps
            cnt[r] += 1; reg.append(L.mean() / P.oracle_cost - 1)
        G /= np.maximum(cnt, 1)
        if st["scale"] is None:
            st["scale"] = np.abs(p - y).mean() / max(np.abs(G[cnt > 0]).mean(), 1e-12)
        G = st["scale"] * G + mu * (p - y)
        H = np.where(cnt > 0, np.mean(p * (1 - p)), 0.0) + mu * p * (1 - p)
        if log is not None: log.append(float(np.mean(reg)))
        st["it"] += 1
        return G, np.maximum(H, 1e-4)
    return fobj

def train_correction(X, y, s0, pools, rows, *, rounds=60, lr=0.05, num_leaves=4, seed=0, **kw):
    log = []
    ds = lgb.Dataset(X, y, init_score=s0, free_raw_data=False)
    params = dict(objective=make_objective(pools, rows, len(y), seed=seed, log=log, **kw), num_leaves=num_leaves,
                  learning_rate=lr, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, min_data_in_leaf=20,
                  lambda_l2=1.0, max_delta_step=1.0, verbose=-1, num_threads=1, seed=seed)
    return lgb.train(params, ds, num_boost_round=rounds), log
