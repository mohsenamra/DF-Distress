# DF-Distress

Code and data for the paper

> M. Amra, *When does decision-focused learning of supplier financial distress pay off? Evidence from risk-averse
> supplier selection with simulated and realised outcomes*.

The paper studies supplier selection and order allocation under correlated supplier distress. Distress
probabilities are plugged into a two-stage, risk-averse (mean-CVaR) sourcing model with backup capacity, correlated
distress and stochastic severity. DF-Distress is a decision-focused correction of a cross-fitted, Platt-calibrated
LightGBM model: a boosted correction is trained on the realised cost of the decisions it induces, with an antithetic
score-function gradient and a cache of exact solutions evaluated in closed form. The repository contains the decision
models, all learning methods and comparators, the experiments on 480 supplier pools from four markets, the
robustness analyses and the evaluation on realised outcomes of Iranian suppliers.

## Contents

```
src/
  ssoa.py                 decision models: M1, two-stage mean-CVaR model M2 (HiGHS), closed-form recourse
  instances.py            supplier pools: prices, capacities, backup fees, severity law
  common.py               data loading, validation splits, accuracy metrics
  baselines.py            LR, LR-CS, XGB, XGB-CS, LGBM, LGBM-Focal with Platt calibration
  dfd.py, run_df.py       surrogate-trained methods: CE, DF-Distress-M1, SPO+, PFYL, DBB
  dfm2.py, run_dfm2.py    DF-Distress and its ablations
  rev_train.py            comparators: capacity-matched log-loss control (CE-cont), SPO+-warm, PFYL-warm,
                          mean-CVaR decision loss (DF-Distress-MC)
  rev_eval.py             decision evaluation of all methods on the 480 test pools (also with S = 200)
  evaluate_plugin.py      plug-in and label-based decision rules
  sensitivity.py, saa.py  alternative decision environments; scenario-size study
  s3_train.py, s3_eval.py robustness: hyper-parameters, strict data protocol, lagged features, alternative label,
                          parameters of the scenario model, misspecified environments, stratified US pools
  real_eval.py            evaluation on realised supplier shortfalls (Iran)
  cal_train.py, cal_eval.py  planner calibrated on realised shortfalls; correction retrained on realised outcomes
  parse_monthly.py        production-volume index from Codal monthly activity reports
  tse_prices.py, mkt_labels.py  market-based distress indicator from TSETMC prices
  run_1_baselines.py ... run_6_realised.py   resumable drivers, one per group of experiments
  make_tables.py, make_figures.py, summarise_results.py   tables and figures
tests/test_models.py      consistency checks of the decision models
tools/                    browser script used to collect the Codal monthly reports
data/                     Iranian panel, realised-outcome data and preparation scripts (see data/README.md)
results/                  out-of-sample predictions and pool-level decision results of all methods
figures/                  figures of the paper
```

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

The public benchmark datasets are not redistributed. Download them as described in `data/README.md` and run
`python data/prepare_public.py`.

## Reproducing the results

All result files of the paper are included in `results/`, so the tables can be produced without re-running the
experiments:

```
python src/summarise_results.py
python src/make_figures.py
```

To re-run the experiments from scratch (each driver is resumable and skips finished folds):

```
python src/run_1_baselines.py      # accuracy-trained models and surrogate-trained benchmarks
python src/run_2_dfdistress.py     # DF-Distress and ablations
python src/run_3_sensitivity.py    # alternative decision environments and SAA study
python src/run_4_comparators.py    # CE-cont, SPO+-warm, PFYL-warm, mean-CVaR loss; evaluation on 480 pools
python src/run_5_robustness.py     # robustness analyses (set S3_WORKERS to the number of cores)
python src/run_6_realised.py       # realised outcomes: calibrated planner and retraining (Iran)
python src/real_eval.py            # realised outcomes with the planner of the paper
```

A full run takes several days on a single 12-core machine; DF-Distress needs 35-56 minutes per fold.

```
python -m pytest tests
```

## Main results

Mean normalised regret (%) over 480 test pools (Iran 140, US 140, Poland 100, Taiwan 100):

| Method | Regret | Reduction by DF-Distress [95% interval] |
|---|---|---|
| DF-Distress | 1.114 | |
| CE-cont (capacity-matched log-loss control) | 1.129 | 1.4 [-4.3, 7.4] |
| CAL-BASE (calibrated base learner) | 1.232 | 9.6 [1.6, 15.6] |
| LGBM-Focal | 1.222 | 8.9 [-0.5, 18.0] |
| SPO+-warm | 1.197 | 7.0 [-4.9, 17.6] |
| Risk-blind rule | 2.581 | 56.9 [51.1, 62.9] |

Intervals come from a fold-cluster bootstrap stratified by dataset. On realised supplier outcomes (Iran, 97 pools),
calibrated probabilities plugged into the two-stage model reduce regret by about 36% relative to the risk-blind rule,
while the decision-focused correction adds no further value.

## License

MIT (code). The Iranian data are compiled from public disclosures on Codal and TSETMC.
