"""SSOA decision models of model_formulation_v3_final.md.
  M1  : risk-neutral spot-only MILP, cost affine in p (training surrogate, Theorems 1–2)
  M2  : two-stage stochastic MILP with backup reservation, correlated distress, stochastic severity, mean-CVaR (SAA)
  eval: realised cost by the closed-form recourse of Lemma 1 (no solver)
"""
import numpy as np
import highspy
from scipy.stats import norm, beta as beta_dist

class Instance:
    def __init__(self, c, K, group, lev, *, D=1.0, f0=0.01, eta_h=0.05, e=1.8, pi=0.10, m=0.05, sbar=0.5,
                 gamma=0.6, L=2, U=5, mu_min=0.2, mu_max=0.8, phi=4.0, delta0=0.02, phi0=50.0):
        self.c = np.asarray(c, float); self.K = np.asarray(K, float); self.N = len(c)
        self.group = np.asarray(group); self.lev = np.asarray(lev, float)
        self.D, self.e, self.pi, self.m, self.sbar, self.gamma, self.L, self.U = D, e, pi, m, sbar, gamma, L, U
        self.f = np.full(self.N, f0); self.h = eta_h * self.c
        self.mu = mu_min + (mu_max - mu_min) * np.clip(np.nan_to_num(self.lev, nan=0.5), 0, 1)
        self.delta0, self.phi, self.phi0 = delta0, phi, phi0
        # M1 affine cost: w_q = a + beta * p
        self.a = self.c + (self.e - self.c) * delta0
        self.beta = (self.e - self.c) * (self.mu - delta0)
        self.u = np.minimum(self.K, self.sbar * self.D)

# ---------------------------------------------------------------- first-stage constraints (shared)
def _add_first_stage(h, ins, with_v=True):
    N = ins.N; inf = highspy.kHighsInf
    # variable order: y[0:N], q[N:2N], v[2N:3N]
    for i in range(N):
        h.addVar(0, 1)
    for i in range(N):
        h.addVar(0, inf)
    for i in range(N):
        h.addVar(0, inf if with_v else 0)
    h.changeColsIntegrality(N, np.arange(N, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * N))
    Y, Q, V = np.arange(N), np.arange(N, 2 * N), np.arange(2 * N, 3 * N)
    h.addRow(ins.D, ins.D, N, Q.astype(np.int32), np.ones(N))                                        # sum q = D
    for i in range(N):
        h.addRow(-inf, 0, 2, np.array([Q[i], Y[i]], np.int32), np.array([1.0, -ins.sbar * ins.D]))   # q <= sbar D y
        h.addRow(0, inf, 2, np.array([Q[i], Y[i]], np.int32), np.array([1.0, -ins.m * ins.D]))       # q >= m D y
        h.addRow(-inf, 0, 3, np.array([Q[i], V[i], Y[i]], np.int32), np.array([1.0, 1.0, -ins.K[i]]))  # q + v <= K y
    for g in np.unique(ins.group):
        idx = Q[ins.group == g]
        h.addRow(-inf, ins.gamma * ins.D, len(idx), idx.astype(np.int32), np.ones(len(idx)))
    h.addRow(ins.L, ins.U, N, Y.astype(np.int32), np.ones(N))
    return Y, Q, V

def _new_highs():
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("mip_rel_gap", 1e-6); h.setOptionValue("presolve", "off")   # presolve off: ~2x faster on M2
    return h

class M1Solver:
    """Build once per instance; re-solve for any cost vector w (y,q) cheaply."""
    def __init__(self, ins):
        self.ins = ins; self.h = _new_highs()
        self.Y, self.Q, self.V = _add_first_stage(self.h, ins, with_v=False)
        self.cache = {}
    def cost_vector(self, p):
        return np.r_[self.ins.f, self.ins.a + self.ins.beta * p]
    def solve_w(self, w):
        N = self.ins.N
        self.h.changeColsCost(2 * N, np.arange(2 * N, dtype=np.int32), np.asarray(w, float))
        self.h.run()
        x = np.array(self.h.getSolution().col_value)
        z = x[:2 * N]; key = tuple(np.round(z, 6)); self.cache[key] = z
        return z
    def solve(self, p):
        return self.solve_w(self.cost_vector(p))
    def best_in_cache(self, W):
        """Solution caching (Mulamba et al. 2021): argmin over cached solutions for each row of W."""
        Z = np.array(list(self.cache.values()))
        return Z[np.argmin(W @ Z.T, axis=1)]

# ---------------------------------------------------------------- scenarios
class ScenarioBank:
    """Common random numbers for distress (two-level Gaussian copula) and severity."""
    def __init__(self, ins, S, rho0, rhog, seed=0):
        rng = np.random.default_rng(seed); N = ins.N
        groups, gi = np.unique(ins.group, return_inverse=True)
        M0 = rng.standard_normal(S); Mg = rng.standard_normal((S, len(groups))); eps = rng.standard_normal((S, N))
        rg = np.full(N, rhog)
        self.A = np.sqrt(rho0) * M0[:, None] + np.sqrt(np.maximum(rg - rho0, 0)) * Mg[:, gi] + np.sqrt(1 - rg) * eps
        self.UB = rng.random((S, N)); self.Ub = rng.random((S, N)); self.ins = ins; self.S = S
    def draw(self, p):
        ins = self.ins
        xi = (self.A <= norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))[None, :]).astype(float)
        a1 = ins.mu * ins.phi; b1 = (1 - ins.mu) * ins.phi
        B = beta_dist.ppf(self.UB, a1[None, :], b1[None, :])
        b = beta_dist.ppf(self.Ub, ins.delta0 * ins.phi0, (1 - ins.delta0) * ins.phi0)
        return xi, xi * B + (1 - xi) * b

def severity_given_xi(ins, xi_real, B, seed=1):
    """theta draws for a realised distress vector xi_real (B draws, common random numbers)."""
    rng = np.random.default_rng(seed); N = ins.N
    UB = rng.random((B, N)); Ub = rng.random((B, N))
    Bd = beta_dist.ppf(UB, (ins.mu * ins.phi)[None, :], ((1 - ins.mu) * ins.phi)[None, :])
    bd = beta_dist.ppf(Ub, ins.delta0 * ins.phi0, (1 - ins.delta0) * ins.phi0)
    return xi_real[None, :] * Bd + (1 - xi_real[None, :]) * bd

# ---------------------------------------------------------------- Lemma 1: closed-form recourse / realised cost
def realised_cost(ins, y, q, v, theta):
    """theta: (B, N). Returns (B,) realised costs."""
    theta = np.atleast_2d(theta)
    first = ins.f @ y + ins.h @ v
    prim = (ins.c[None, :] * (1 - theta) * q[None, :]).sum(1)
    short = (theta * q[None, :]).sum(1)
    avail = (1 - theta) * v[None, :]
    order = np.argsort(ins.c); price = (1 + ins.pi) * ins.c[order]
    cum = np.cumsum(avail[:, order], axis=1)
    used_before = np.concatenate([np.zeros((len(short), 1)), cum[:, :-1]], axis=1)
    use = np.clip(short[:, None] - used_before, 0, avail[:, order])
    rec = (use * price[None, :]).sum(1) + ins.e * np.clip(short - cum[:, -1], 0, None)
    return first + prim + rec

# ---------------------------------------------------------------- M2
class M2Solver:
    def __init__(self, ins, bank, lam=0.3, alpha=0.9):
        self.ins, self.bank, self.lam, self.alpha = ins, bank, lam, alpha
    def solve_theta(self, th, return_obj=False):
        """Solve for a given scenario matrix theta (S,N) — used by the ex-post oracle."""
        return self.solve(None, return_obj, theta=th)
    def solve(self, p, return_obj=False, theta=None):
        ins, N, lam, al = self.ins, self.ins.N, self.lam, self.alpha
        th = theta if theta is not None else self.bank.draw(p)[1]
        S = th.shape[0]
        h = _new_highs(); inf = highspy.kHighsInf
        Y, Q, V = _add_first_stage(h, ins, with_v=True)
        base = 3 * N
        R = base + np.arange(S * N).reshape(S, N); Uv = base + S * N + np.arange(S)
        ETA = base + S * N + S; TAU = ETA + 1 + np.arange(S)
        for _ in range(S * N + S):
            h.addVar(0, inf)
        h.addVar(-inf, inf)
        for _ in range(S):
            h.addVar(0, inf)
        w = (1 - lam) / S
        cost = np.zeros(TAU[-1] + 1)
        cost[Y] = ins.f; cost[V] = ins.h
        cost[Q] = w * (ins.c[None, :] * (1 - th)).sum(0)
        cost[R] = w * (1 + ins.pi) * ins.c[None, :]
        cost[Uv] = w * ins.e
        cost[ETA] = lam; cost[TAU] = lam / ((1 - al) * S)
        # first-stage parts of C^s appear in the CVaR rows only through tau; expectation part handled via cost above
        # but f, h are scenario-independent: the objective (1-lam)E[C] + lam CVaR[C] counts them once in each term
        cost[Y] = ins.f; cost[V] = ins.h   # (1-lam)*f + lam*f  (CVaR is translation-equivariant)
        h.changeColsCost(len(cost), np.arange(len(cost), dtype=np.int32), cost)
        for s in range(S):
            # sum_i r_is + u_s - sum_i theta_is q_i = 0
            idx = np.r_[R[s], Uv[s], Q].astype(np.int32); val = np.r_[np.ones(N), 1.0, -th[s]]
            h.addRow(0, 0, len(idx), idx, val)
            for i in range(N):   # r_is <= (1-theta_is) v_i
                h.addRow(-inf, 0, 2, np.array([R[s, i], V[i]], np.int32), np.array([1.0, -(1 - th[s, i])]))
            # tau_s >= C^s - eta, with C^s excluding the constant first-stage costs (translation equivariance)
            idx = np.r_[TAU[s], ETA, Q, R[s], Uv[s]].astype(np.int32)
            val = np.r_[1.0, 1.0, -ins.c * (1 - th[s]), -(1 + ins.pi) * ins.c, -ins.e]
            h.addRow(0, inf, len(idx), idx, val)
        h.run()
        x = np.array(h.getSolution().col_value)
        out = (x[Y].round(), x[Q], x[V])
        if return_obj:
            return out + (h.getInfo().objective_function_value,)
        return out

def m1_decision_to_yqv(ins, z):
    N = ins.N
    return z[:N].round(), z[N:2 * N], np.zeros(N)

def mean_cvar(C, lam=0.3, alpha=0.9):
    """(1-lam) mean + lam CVaR_alpha of equally weighted scenario costs (Rockafellar-Uryasev, exact for SAA)."""
    C = np.asarray(C, float); eta = np.quantile(C, alpha, method="inverted_cdf")
    return (1 - lam) * C.mean() + lam * (eta + np.maximum(C - eta, 0).mean() / (1 - alpha))

# ---------------------------------------------------------------- M2 by exact dual-cut reformulation (Lemma 1)
def recourse_value_and_dual(ins, q, v, theta):
    """Per-scenario recourse Q_s and the greedy dual price lambda_s (marginal source price). theta: (S,N)."""
    theta = np.atleast_2d(theta)
    short = theta @ q; avail = (1 - theta) * v[None, :]
    order = np.argsort(ins.c); price = (1 + ins.pi) * ins.c[order]
    cum = np.cumsum(avail[:, order], axis=1)
    used_before = np.concatenate([np.zeros((len(short), 1)), cum[:, :-1]], axis=1)
    use = np.clip(short[:, None] - used_before, 0, avail[:, order])
    spot = np.clip(short - cum[:, -1], 0, None)
    Qs = (use * price[None, :]).sum(1) + ins.e * spot
    # dual price: e if spot used; else price of last (partially) used backup; else 0 when shortfall is nil
    lam = np.zeros(len(short))
    k_last = np.where(use > 1e-12, np.arange(ins.N)[None, :], -1).max(1)
    lam[k_last >= 0] = price[k_last[k_last >= 0]]
    lam[spot > 1e-12] = ins.e
    return Qs, lam

class M2CutSolver:
    """Same optimisation problem as M2Solver, with recourse replaced by its value function
    Q_s(q,v) = max_{lambda in Lambda} [ lambda * theta_s.q - sum_i (lambda-(1+pi)c_i)^+ (1-theta_is) v_i ],
    Lambda = {0, (1+pi)c_i, e}  (dual vertices of the greedy recourse, Lemma 1). Cuts are added lazily."""
    def __init__(self, ins, bank=None, lam=0.3, alpha=0.9, tol=1e-6, max_iter=40):
        self.ins, self.bank, self.lam, self.alpha, self.tol, self.max_iter = ins, bank, lam, alpha, tol, max_iter
        self.iters = 0
    def _cut(self, lamv, th_s):
        ins = self.ins
        coef_q = lamv * th_s
        coef_v = -np.maximum(lamv - (1 + ins.pi) * ins.c, 0) * (1 - th_s)
        return coef_q, coef_v
    def solve(self, p=None, return_obj=False, theta=None):
        ins, N, lam, al = self.ins, self.ins.N, self.lam, self.alpha
        th = theta if theta is not None else self.bank.draw(p)[1]
        S = th.shape[0]; inf = highspy.kHighsInf
        h = _new_highs()
        Y, Q, V = _add_first_stage(h, ins, with_v=True)
        base = 3 * N; PHI = base + np.arange(S); ETA = base + S; TAU = ETA + 1 + np.arange(S)
        for _ in range(S): h.addVar(0, inf)
        h.addVar(-inf, inf)
        for _ in range(S): h.addVar(0, inf)
        w = (1 - lam) / S
        cost = np.zeros(TAU[-1] + 1)
        cost[Y] = ins.f; cost[V] = ins.h
        cost[Q] = w * (ins.c[None, :] * (1 - th)).sum(0)
        cost[PHI] = w; cost[ETA] = lam; cost[TAU] = lam / ((1 - al) * S)
        h.changeColsCost(len(cost), np.arange(len(cost), dtype=np.int32), cost)
        for s in range(S):   # tau_s + eta - c(1-theta_s).q - phi_s >= 0
            idx = np.r_[TAU[s], ETA, Q, PHI[s]].astype(np.int32)
            h.addRow(0, inf, len(idx), idx, np.r_[1.0, 1.0, -ins.c * (1 - th[s]), -1.0])
        def add_cuts(lams):
            for s, lv in lams:
                cq, cv = self._cut(lv, th[s])
                idx = np.r_[PHI[s], Q, V].astype(np.int32)
                h.addRow(0, inf, len(idx), idx, np.r_[1.0, -cq, -cv])
        add_cuts([(s, ins.e) for s in range(S)])           # spot-price cut (valid, tight when no backup)
        best = (np.inf, None)
        for it in range(self.max_iter):
            h.run()
            x = np.array(h.getSolution().col_value)
            LB = h.getInfo().objective_function_value
            y, q, v = x[Y].round(), x[Q], x[V]
            Qs, lamv = recourse_value_and_dual(ins, q, v, th)
            UB = mean_cvar(ins.f @ y + ins.h @ v + ((1 - th) * ins.c[None, :]) @ q + Qs, lam, al)
            if UB < best[0]:
                best = (UB, (y, q, v))
            self.iters, self.gap = it + 1, (best[0] - LB) / abs(best[0])
            if self.gap <= self.tol:
                break
            viol = Qs - x[PHI] > 1e-9
            add_cuts([(s, lamv[s]) for s in np.where(viol)[0]])
        y, q, v = best[1]; out = (y, q, v)
        out = (x[Y].round(), q, v)
        if return_obj:
            return out + (h.getInfo().objective_function_value,)
        return out
