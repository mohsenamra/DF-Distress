# Data

## Iranian panel (included)

Compiled from the audited annual financial statements of non-financial companies listed on the Tehran Stock Exchange
and Iran Fara Bourse, as published on Codal (codal.ir), fiscal years 1393-1404 of the Iranian calendar. Delisted
firms are included. Parent-only statements are used. Gaps in the structured balance sheets were filled from the
comparative columns of later audited statements and from six-month interim reports; such values are flagged.

| File | Content |
|---|---|
| `panel_items_merged.csv` | standardised statement items per firm-year, with source flags (`filled_from_next`, `filled_from_later`, `bs_from_interim`) |
| `dataset.csv` | features at year t and labels at year t+1 (output of `build_features.py`) |
| `dataset_model.csv` | modelling file: CPI-deflated size and growth, winsorised features (`*_w`) (output of `prepare_data.py`) |
| `supplier_pool.csv` | firm-years in the 14 supply industries |
| `cpi_iran.csv` | urban consumer price index, Statistical Centre of Iran (1400 = 100) |
| `copula_params.json` | estimated global and within-industry distress correlation |
| `market_labels.csv` | market-based distress indicator per firm-year (output of `src/mkt_labels.py`) |
| `codal_monthly/monthly_reports.json` | production and sales tables of the year-end monthly activity reports on Codal |
| `codal_monthly/volume_index.csv` | Laspeyres production and sales volume indices per firm-year (output of `src/parse_monthly.py`) |

Amounts are in million rials as reported. Main label `y_a141`: accumulated losses at least 50% of registered capital
at the fiscal year-end t+1 (Article 141 of the Iranian Commercial Code). Alternative labels: `y_negeq` (negative
equity at t+1) and `y_twoloss` (net loss in t and t+1). `a141_t` is the status at t.

Rebuild the modelling files from the panel with

```
python data/build_features.py
python data/prepare_data.py
```

## Realised supplier outcomes

The realised delivery shortfall of a supplier is the loss of CPI-deflated revenue in the outcome year,
`theta = clip(1 - R[t+1] / R[t], 0, 1)`, computed in `src/real_eval.py` from `dataset_model.csv`. It is validated
against the production-volume index built from the year-end monthly activity reports:

```
python src/parse_monthly.py data/codal_monthly/monthly_reports.json   # writes volume_index.csv in the working directory
```

The monthly reports were collected in a browser session on codal.ir with `tools/codal_monthly_browser.js`, at a rate
that respects the access limits of the site.

The market-based distress indicator uses adjusted daily prices from TSETMC (tsetmc.com). A firm-year is in market
distress if the adjusted price falls by more than 50% over the outcome year or the shares are not traded for 120
consecutive days or more:

```
python src/tse_prices.py     # downloads data/tse_prices/ (not included)
python src/mkt_labels.py     # writes data/market_labels.csv
```

## Public datasets (not included)

Download the original files into `data/public/` and run `python data/prepare_public.py`:

| Dataset | Source | Files |
|---|---|---|
| Poland | UCI Machine Learning Repository, *Polish companies bankruptcy data* (Tomczak, 2016; Zięba et al., 2016) | `1year.arff` ... `5year.arff` |
| Taiwan | UCI Machine Learning Repository, *Taiwanese bankruptcy prediction* (Liang et al., 2016) | `taiwan_bankruptcy.csv` |
| United States | Lombardo et al. (2022), *Future Internet* 14(8), 244 | `american_bankruptcy.csv` |

The script writes `std_polish.csv`, `std_taiwan.csv` and `std_us.csv`, which `src/common.py` reads.
