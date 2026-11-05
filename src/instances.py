"""Supplier-pool instances for SSOA experiments (model_formulation_v3_final.md, §4b, §7).

An instance = N candidate suppliers observed in the same decision year t (same fold), with
  - realised distress xi_i = y_i (label at t+1),
  - price  c_i = cbar (1 - kappa_p * r_i + eps_i),  r_i = rank_[0,1] of mean rank(TL/TA, -ROA, -gross margin),
  - capacity K_i = D * clip(kappa_K * sqrt(size_i)/median(sqrt(size)), K_min, K_max)  (size = real revenue, or TA),
  - leverage (for severity mean mu_i) = TL/TA,
  - industry group g(i) (real for IR/US; synthetic 5 groups for PL/TW -> semi-synthetic pools).
Pools are drawn without replacement inside a fold's test set; seeds make pools and eps common to all methods.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")

import numpy as np, pandas as pd
from scipy.stats import rankdata
from ssoa import Instance

DATA = ROOT + "/data/"

def _rank01(x):
    x = np.asarray(x, float); x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0)
    return (rankdata(x) - 1) / max(len(x) - 1, 1)

def attributes(name):
    """Return DataFrame keyed by dataset row index (as in common.load) with lev, roa, gm, size, group."""
    from common import load
    d, _ = load(name)
    if name == "IR":
        raw = pd.read_csv(DATA + "dataset_model.csv")
        raw = raw[raw.y_a141.notna()].reset_index(drop=True)          # same row order as load("IR")
        A = pd.DataFrame(dict(lev=raw.tl_ta_w, roa=raw.roa_w, gm=raw.gross_margin_w,
                              size=raw.revenue_t / raw.cpi * 100, group=raw.industry, pool=raw.supply_pool))
    elif name == "US":
        A = pd.DataFrame(dict(lev=d.tl_ta, roa=d.roa, gm=d.gross_margin, size=np.exp(d.log_rev), group=d.group, pool=1))
    elif name == "PL":
        A = pd.DataFrame(dict(lev=d.a2, roa=d.a1, gm=d.a19, size=np.exp(d.a29.clip(-5, 12)), group=np.nan, pool=1))
    elif name == "TW":
        A = pd.DataFrame(dict(lev=d.t37, roa=d.t2, gm=d.t4, size=np.nan, group=np.nan, pool=1))
    A.index = d.index
    return A

def make_pool_instance(A_sub, rng, *, kappa_p=0.10, sigma_c=0.03, cbar=1.0, kappa_K=0.35, K_min=0.10, K_max=1.0,
                       n_syn_groups=5, **ins_kw):
    N = len(A_sub)
    r = _rank01((_rank01(A_sub.lev) + _rank01(-A_sub.roa.values) + _rank01(-A_sub.gm.values)) / 3)
    eps = rng.normal(0, sigma_c, N)
    c = cbar * (1 - kappa_p * r + eps)
    size = A_sub["size"].values.astype(float)
    if not np.isfinite(size).all() or (size <= 0).any():
        s = np.where(np.isfinite(size) & (size > 0), size, np.nan)
        fill = rng.lognormal(0, 1, N) * (np.nanmedian(s) if np.isfinite(s).any() else 1.0)
        size = np.where(np.isfinite(s), s, fill)
    sq = np.sqrt(size); K = np.clip(kappa_K * sq / np.median(sq), K_min, K_max)
    g = A_sub.group.values
    if pd.isna(g).any():
        g = rng.integers(0, n_syn_groups, N)
    else:
        g = pd.factorize(g)[0]
    lev = np.clip(np.nan_to_num(A_sub.lev.values.astype(float), nan=0.5), 0, 1)
    return Instance(c, K, g, lev, **ins_kw)

def draw_pools(test_idx, A, n_pools, N=20, seed=0, min_groups=3):
    """Indices of n_pools pools of size N from test_idx (pool-eligible rows), each spanning >= min_groups groups."""
    rng = np.random.default_rng(seed)
    elig = np.array([i for i in test_idx if A.pool.get(i, 1) == 1])
    if len(elig) < N:
        return []
    pools = []
    for _ in range(n_pools * 20):
        sel = rng.choice(elig, N, replace=False)
        gs = A.loc[sel, "group"]
        if gs.isna().all() or gs.nunique() >= min_groups:
            pools.append(np.sort(sel))
        if len(pools) == n_pools:
            break
    return pools
