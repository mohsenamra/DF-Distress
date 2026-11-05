"""Evaluation of the trained policies against realised supplier outcomes (Iran).

The decisions are planned exactly as in the paper (M2 with the scenario model of Section 3), on the test pools of
rev_eval.py (same pools and seeds). Instead of drawing the delivery shortfall of each supplier from the severity
model, the realised shortfall of supplier i in the outcome year is taken from the data:
  rev     theta_i = clip(1 - R_{i,t+1} / R_{i,t}, 0, 1), R = CPI-deflated revenue from the audited statements
  vol     theta_i = clip(1 - I_{i,t+1}, 0, 1), I = Laspeyres production-volume index from the year-end Codal
          monthly activity report (data/codal_monthly/volume_index.csv); suppliers without a report keep the
          revenue-based value
The realised cost and the ex-post oracle are computed for that single realisation. Pools are kept when at least
18 of the 20 suppliers have a realised value; missing ones are filled with the mean realised shortfall of suppliers
with the same distress outcome in the same fold.
Output: results/real_eval_IR.csv (fold, pool, outcome, model, nregret, n_real, n_distress)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")
import sys, time, zlib, numpy as np, pandas as pd
from common import load
from instances import attributes, make_pool_instance, draw_pools
from ssoa import M2Solver, ScenarioBank, realised_cost
from run_dfm2 import splits_for, RHO

OUT = ROOT + "/results/"
S, NPOOL, N = 100, int(_os.environ.get("NPOOL", 20)), 20
MODELS = (("CAL-BASE", "_dfm2"), ("DFM2", "_dfm2"), ("DFM2-max", "_dfm2"), ("CE-cont", "_rev"), ("SPO+-warm", "_rev"),
          ("PFYL-warm", "_rev"), ("LGBM-Focal", ""), ("LGBM", ""), ("XGB", ""))


def realised_shortfall(d):
    raw = pd.read_csv(ROOT + "/data/dataset_model.csv")
    raw = raw[raw.y_a141.notna()].reset_index(drop=True)            # same row order as load("IR")
    rr = raw.revenue_t / raw.cpi
    key = raw.symbol.str.strip() + "|" + raw.fy.astype(int).astype(str)
    nxt = pd.Series(rr.values, index=key)
    nk = raw.symbol.str.strip() + "|" + (raw.fy.astype(int) + 1).astype(str)
    ratio = nk.map(nxt).values / rr.values
    th_rev = np.clip(1 - ratio, 0, 1); th_rev[~np.isfinite(ratio) | (ratio <= 0.02) | (ratio > 10)] = np.nan
    th_vol = th_rev.copy()
    fn = ROOT + "/data/codal_monthly/volume_index.csv"
    if _os.path.exists(fn):
        v = pd.read_csv(fn); v = v[(v.prod_index > 0.05) & (v.prod_index < 3)]
        vi = pd.Series(v.prod_index.values, index=v.symbol.str.strip() + "|" + v.fy.astype(int).astype(str))
        iv = nk.map(vi).values
        ok = np.isfinite(iv); th_vol[ok] = np.clip(1 - iv[ok], 0, 1)
        has_vol = ok
    else:
        has_vol = np.zeros(len(raw), bool)
    return pd.DataFrame(dict(rev=th_rev, vol=th_vol, has_vol=has_vol), index=d.index)


def run():
    name = "IR"; d, _ = load(name); A = attributes(name); rho0, rhog = RHO[name]
    TH = realised_shortfall(d)
    P = {}
    for _, sfx in MODELS:
        if sfx not in P:
            t = pd.read_csv(OUT + f"preds_{name}{sfx}.csv.gz", usecols=["fold", "model", "idx", "p_cal"])
            t["fold"] = t.fold.astype(str); P[sfx] = t
    rows = []
    for fold, tr, te in splits_for(name, d):
        fold = str(fold); t0 = time.time(); key = zlib.crc32(f"{name}{fold}".encode())
        base = float(d.loc[tr, "y"].mean())
        probs = {m: P[s][(P[s].fold == fold) & (P[s].model == m)].set_index("idx").p_cal for m, s in MODELS}
        fill = {o: {y: np.nanmean(TH.loc[te][d.loc[te, "y"] == y][o]) for y in (0, 1)} for o in ("rev", "vol")}
        for k, idx in enumerate(draw_pools(te, A, NPOOL, N, seed=key)):
            xi = d.loc[idx, "y"].values
            n_real = int(np.isfinite(TH.loc[idx, "rev"].values).sum())
            if n_real < N - 2: continue
            rng = np.random.default_rng([key, k]); ins = make_pool_instance(A.loc[idx], rng)
            m2 = M2Solver(ins, ScenarioBank(ins, S, rho0, rhog, seed=k + 11))
            dec = {"NAIVE": m2.solve(np.full(N, base))}
            for m in probs: dec[m] = m2.solve(probs[m].loc[idx].values)
            for o in ("rev", "vol"):
                th = TH.loc[idx, o].values.copy()
                miss = ~np.isfinite(th); th[miss] = [fill[o][int(y)] for y in xi[miss]]
                th = th[None, :]
                orc = float(realised_cost(ins, *M2Solver(ins, None, lam=0.0).solve_theta(th), th).mean())
                for m, z in dec.items():
                    c = float(realised_cost(ins, *z, th).mean())
                    rows.append(dict(fold=fold, pool=k, outcome=o, model=m, nregret=(c - orc) / orc, n_real=n_real,
                                     n_vol=int(TH.loc[idx, "has_vol"].sum()), n_distress=int(xi.sum()),
                                     mean_theta=float(th.mean())))
        pd.DataFrame(rows).to_csv(OUT + "real_eval_IR.csv", index=False)
        print(name, fold, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
