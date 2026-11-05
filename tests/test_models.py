"""Consistency checks of the decision models: run with  python -m pytest tests  (or python tests/test_models.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ssoa import Instance, M2Solver, M2CutSolver, ScenarioBank, realised_cost, mean_cvar, recourse_value_and_dual


def _instance(seed=3, N=20):
    rng = np.random.default_rng(seed)
    lev = rng.uniform(0.3, 1.1, N); c = rng.normal(1.0, 0.05, N)
    K = rng.uniform(0.15, 0.5, N); group = rng.integers(0, 5, N)
    p = np.clip(rng.beta(0.6, 6, N), 0.002, 0.9)
    return Instance(c, K, group, lev), p


def test_lemma1_matches_milp_objective():
    """The SAA objective reported by the MILP equals the closed-form (Lemma 1) mean-CVaR of its own solution."""
    ins, p = _instance()
    bank = ScenarioBank(ins, 100, 0.054, 0.122, seed=0)
    th = bank.draw(p)[1]
    y, q, v, obj = M2Solver(ins, bank).solve(None, return_obj=True, theta=th)
    assert abs(mean_cvar(realised_cost(ins, y, q, v, th), 0.3, 0.9) - obj) < 1e-6 * abs(obj)


def test_cut_reformulation_is_equivalent():
    """Extensive form (Eq. 10) and dual-vertex cut reformulation (Eq. 9) give the same optimal value."""
    for seed in range(5):
        ins, p = _instance(seed)
        bank = ScenarioBank(ins, 100, 0.054, 0.122, seed=seed)
        th = bank.draw(p)[1]
        f = lambda z: mean_cvar(realised_cost(ins, *z, th), 0.3, 0.9)
        z1 = M2Solver(ins, bank).solve(None, theta=th); z2 = M2CutSolver(ins, bank).solve(theta=th)
        assert abs(f(z1) - f(z2)) < 1e-6 * f(z1)


def test_recourse_is_max_of_affine_functions():
    """Lemma 1: the greedy recourse equals the maximum over the dual vertices in Lambda."""
    ins, p = _instance()
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(ins.N)); v = rng.uniform(0, 0.1, ins.N); th = rng.uniform(0, 1, (50, ins.N))
    R, _ = recourse_value_and_dual(ins, q, v, th)
    lams = np.r_[0.0, (1 + ins.pi) * ins.c, ins.e]
    for s in range(th.shape[0]):
        vals = [l * th[s] @ q - np.maximum(l - (1 + ins.pi) * ins.c, 0) @ ((1 - th[s]) * v) for l in lams]
        assert abs(max(vals) - R[s]) < 1e-9


if __name__ == "__main__":
    test_lemma1_matches_milp_objective(); test_cut_reformulation_is_equivalent(); test_recourse_is_max_of_affine_functions()
    print("all tests passed")
