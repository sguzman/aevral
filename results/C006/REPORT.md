# C006 final report — trees and boosted ensembles
Complete: 28/28 screens, 124/124 additional outcomes, 152 unique measured results, zero blocks. All 17 native and 14 upstream annual jobs passed; 95 numerical tests. 20 finalists plus reused C003 incumbent. 30,138 matched annual origins and 723,312 targets per seed. No new data or 2023+ tuning.

## Pooled ranking
MAE weighted by fold origin count; RMSE square root of weighted squared fold RMSE. Pool within each seed, then arithmetic mean and sample SD (n−1) across three seeds for stochastic configurations. Single-seed SD not estimated. Rank by mean pooled MAE then RMSE. Seed replicates are not independent additional weather observations. Computed from archived rounded aggregate scores; not independently rescored from raw prediction artifacts.

| Rank | Model | MAE °C | Seed SD MAE | RMSE °C | Seed SD RMSE |
|---:|---|---:|---:|---:|---:|
| 1 | mlp64 | 1.34781461 | — | 1.80956416 | — |
| 2 | lightgbm-128 | 1.40602200 | — | 1.89032671 | — |
| 3 | xgboost-128 | 1.41090543 | — | 1.89795122 | — |
| 4 | hist-64-d5 | 1.45552862 | — | 1.94599550 | — |
| 5 | lightgbm-64 | 1.45688740 | — | 1.95327573 | — |
| 6 | xgboost-64 | 1.46193109 | — | 1.96107605 | — |
| 7 | catboost-128 | 1.47080423 | 0.00189789 | 1.96237987 | 0.00238772 |
| 8 | catboost-64 | 1.55729717 | 0.00285304 | 2.06886342 | 0.00339067 |
| 9 | hist-32-d5 | 1.56193337 | — | 2.07510744 | — |
| 10 | bag-32 | 1.60511107 | 0.00198223 | 2.12833670 | 0.00197705 |
| 11 | median-boost-64 | 1.60623564 | — | 2.14407312 | — |
| 12 | bag-16 | 1.61266862 | 0.00380988 | 2.13770834 | 0.00428441 |
| 13 | rf-32-27 | 1.63294832 | 0.00117453 | 2.15714625 | 0.00159269 |
| 14 | rf-16-27 | 1.64345583 | 0.00075545 | 2.17037379 | 0.00057687 |
| 15 | cart-lead-d8-l32 | 1.67195161 | — | 2.23232089 | — |
| 16 | extra-32-27 | 1.75068713 | 0.00091107 | 2.28862232 | 0.00094126 |
| 17 | extra-16-27 | 1.76079905 | 0.00247715 | 2.30179818 | 0.00356714 |
| 18 | cart-shared-d8-l32 | 1.79494569 | — | 2.36770125 | — |
| 19 | cart-shared-d8-l128 | 1.85887917 | — | 2.45599742 | — |
| 20 | median-boost-32 | 1.87096974 | — | 2.55667775 | — |
| 21 | cart-lead-d4-l32 | 1.92806447 | — | 2.53238952 | — |

Best tree lightgbm-128: MAE 1.40602200, 4.32% above MLP64 1.34781461. Tree models use 24,576 evenly spaced training origins per annual fold, MLP64 uses the full causal training cohort (121,849–144,417). Same evaluation cohorts and input schema, unequal training budgets. No claim of equal-data superiority. XGBoost and LightGBM close; no significance test for their difference. Median boosting fits the 0.5 quantile; broader uncertainty evaluation deferred to C014 as specified in manifest.

## Reproducibility
[Registry](registry.csv), [pooled scores](pooled.csv), [per-seed scores](pooled-seeds.csv), [finalists](../../campaigns/006-trees-boosting/FINALISTS.md), [native CI](https://github.com/sguzman/aevral/actions/runs/45917952848), [upstream CI](https://github.com/sguzman/aevral/actions/runs/45918425331). Official backends XGBoost 3.0.5, LightGBM 4.6.0, CatBoost 1.2.10 pinned in Nix. No backend substitutions. Per-lead/month diagnostics and resource logs archived. Raw prediction artifacts have 30-day retention; aggregate rankings are durable. Failed first CatBoost parser smoke archived and fixed, all official screens and finalist evaluations passed. C007 next.
