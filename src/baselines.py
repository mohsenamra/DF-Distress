"""Step 5a — accuracy-trained and cost-sensitive prediction baselines.
Models: LR (L2), LR-CS (class-weighted), XGB, XGB-CS (scale_pos_weight), LGBM, LGBM-Focal (imbalance-aware boosting,
focal loss; our stand-in for imbalance-aware GBDT such as PROBoost, which has no public code).
Each model is calibrated (Platt scaling) on an inner validation split and the G-mean threshold is tuned there
(the 'threshold-tuned' variant used later by label-based decision rules).
Validation: rolling origin (IR, US) or 5-fold stratified CV x 2 repeats (PL, TW).
Output: results/baselines_{name}.csv (per fold) and results/preds_{name}.parquet (out-of-sample probabilities).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace("\\", "/")

import sys, os, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _Platt
import xgboost as xgb, lightgbm as lgb
from common import load, rolling_splits, cv_splits, metrics, best_gmean_threshold
warnings.filterwarnings("ignore")
OUT = ROOT + "/results/"; os.makedirs(OUT, exist_ok=True)

def focal_obj(gamma=2.0, alpha=0.25):
    def f(preds, data):
        y = data.get_label(); p = 1 / (1 + np.exp(-preds)); eps = 1e-9
        # gradient/hessian of focal loss w.r.t. raw score (numerical, stable)
        def loss(z):
            pp = 1 / (1 + np.exp(-z)); pt = np.where(y == 1, pp, 1 - pp); at = np.where(y == 1, alpha, 1 - alpha)
            return -at * (1 - pt) ** gamma * np.log(np.clip(pt, eps, 1))
        h = 1e-3
        g = (loss(preds + h) - loss(preds - h)) / (2 * h)
        hs = (loss(preds + h) - 2 * loss(preds) + loss(preds - h)) / h ** 2
        return g, np.maximum(hs, 1e-6)
    return f

def fit_predict(model, Xtr, ytr, Xte):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    if model in ("LR", "LR-CS"):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced" if model == "LR-CS" else None))
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1], m
    if model in ("XGB", "XGB-CS"):
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=2, reg_lambda=1.0, scale_pos_weight=spw if model == "XGB-CS" else 1.0,
                              eval_metric="logloss", n_jobs=2, verbosity=0)
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1], m
    if model == "LGBM":
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, min_child_samples=10, verbose=-1, n_jobs=2)
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1], m
    if model == "LGBM-Focal":
        ds = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        params = dict(objective=focal_obj(), num_leaves=15, learning_rate=0.05, bagging_fraction=0.8, bagging_freq=1,
                      feature_fraction=0.8, min_data_in_leaf=10, verbose=-1, num_threads=2)
        b = lgb.train(params, ds, num_boost_round=300)
        return 1 / (1 + np.exp(-b.predict(Xte, raw_score=True))), b
    raise ValueError(model)

MODELS = ["LR", "LR-CS", "XGB", "XGB-CS", "LGBM", "LGBM-Focal"]

def run(name):
    d, feats = load(name)
    if name == "IR":
        splits = list(rolling_splits(d, range(1397, 1404)))
    elif name == "US":
        splits = list(rolling_splits(d, range(2011, 2018)))
    else:
        splits = [(f"r{r}k{k}", tr, te) for r in range(2) for k, tr, te in cv_splits(d, 5, seed=r)]
    rows, preds = [], []
    for fold, tr, te in splits:
        dtr = d.loc[tr]; dte = d.loc[te]
        # inner validation: last training year (temporal) or a 20% stratified hold-out (CV datasets)
        if name in ("IR", "US"):
            last = dtr.year.max(); itr = dtr.index[dtr.year < last]; iva = dtr.index[dtr.year == last]
            if d.loc[iva, "y"].sum() < 3:
                iva = dtr.index[dtr.year >= last - 1]; itr = dtr.index[dtr.year < last - 1]
        else:
            rng = np.random.default_rng(0); msk = rng.random(len(dtr)) < 0.2
            itr, iva = dtr.index[~msk], dtr.index[msk]
        for mname in MODELS:
            t0 = time.time()
            p_va, _ = fit_predict(mname, d.loc[itr, feats].values, d.loc[itr, "y"].values, d.loc[iva, feats].values)
            lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            platt = _Platt(C=1e6).fit(lo(p_va), d.loc[iva, "y"].values)   # Platt scaling on the inner validation split
            thr = best_gmean_threshold(d.loc[iva, "y"].values, platt.predict_proba(lo(p_va))[:, 1])
            p_raw, _ = fit_predict(mname, dtr[feats].values, dtr.y.values, dte[feats].values)
            p_cal = platt.predict_proba(lo(p_raw))[:, 1]
            for variant, p in (("raw", p_raw), ("cal", p_cal)):
                m = metrics(dte.y.values, p, thr if variant == "cal" else best_gmean_threshold(d.loc[iva, "y"].values, p_va))
                rows.append(dict(dataset=name, fold=str(fold), model=mname, variant=variant, n_test=len(dte),
                                 pos_test=int(dte.y.sum()), thr=thr, secs=time.time() - t0, **m))
            preds.append(pd.DataFrame(dict(dataset=name, fold=str(fold), model=mname, idx=te, firm=dte.firm.values,
                                           year=dte.year.values, y=dte.y.values, p_raw=p_raw, p_cal=p_cal, thr=thr)))
        print(name, fold, "done", flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT + f"baselines_{name}.csv", index=False)
    pd.concat(preds).to_csv(OUT + f"preds_{name}.csv.gz", index=False)
    s = r.groupby(["model", "variant"])[["auc", "prauc", "brier", "logloss", "gmean"]].mean().round(4)
    print(s.to_string())

if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
