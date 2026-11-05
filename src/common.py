"""Shared data loading, splits and metrics for DF-Distress experiments."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

DATA = ROOT + "/data/"

IRAN_FEATURES = ["current_ratio_w", "quick_ratio_w", "cash_ta_w", "wc_ta_w", "tl_ta_w", "debt_ta_w", "st_debt_ta_w",
                 "eq_ta_w", "fincost_rev_w", "int_cover_w", "roa_w", "ebit_ta_w", "gross_margin_w", "op_margin_w",
                 "net_margin_w", "loss_t", "rev_ta_w", "inv_ta_w", "rec_ta_w", "pay_ta_w", "ppe_ta_w", "cfo_ta_w",
                 "cfo_tl_w", "capex_ta_w", "log_ta_real_w", "log_rev_real_w", "rev_growth_real_w", "ta_growth_real_w",
                 "loss_prev", "a141_t"]

def load(name):
    """Return df with columns: firm, year, group, y, features...; and the feature list."""
    if name == "IR":
        d = pd.read_csv(DATA + "dataset_model.csv")
        d = d.rename(columns={"symbol": "firm", "fy": "year", "industry": "group", "y_a141": "y"})
        feats = IRAN_FEATURES
    else:
        f = {"US": "std_us.csv", "PL": "std_polish.csv", "TW": "std_taiwan.csv"}[name]
        fp = DATA + "public/" + f
        d = pd.read_csv(fp if _os.path.exists(fp) else fp + ".gz")
        feats = [c for c in d.columns if c not in ("dataset", "firm", "year", "group", "y", "re_ta")]
        # winsorise public features (1/99%) — Taiwan is already bounded
        for c in feats:
            if d[c].nunique() > 2:
                lo, hi = d[c].quantile([0.01, 0.99]); d[c] = d[c].clip(lo, hi)
    d = d[d.y.notna()].reset_index(drop=True)
    d["y"] = d.y.astype(int)
    return d, feats

def rolling_splits(d, test_years):
    """Train on firm-years with year <= tau-1 (labels realised by tau), test on year tau."""
    for tau in test_years:
        tr = d.index[d.year <= tau - 1]; te = d.index[d.year == tau]
        if len(te) and d.loc[tr, "y"].sum() >= 10:
            yield tau, tr, te

def cv_splits(d, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(d, d.y)):
        yield k, d.index[tr], d.index[te]

def gmean_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tpr = ((yhat == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = ((yhat == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float(np.sqrt(tpr * tnr))

def metrics(y, p, thr):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(auc=roc_auc_score(y, p), prauc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                logloss=log_loss(y, p, labels=[0, 1]), gmean=gmean_at(y, p, thr), base_rate=float(np.mean(y)))

def best_gmean_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 120)))
    g = [gmean_at(y, p, t) for t in qs]
    return float(qs[int(np.argmax(g))])
