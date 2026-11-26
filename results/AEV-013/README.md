# AEV-013 compact attention
[Successful CI](https://github.com/sguzman/aevral/actions/runs/45877542890), code f542af728a525092fe5053377a13255b6257ade5. Same 121849 train and 30207 validation origins.
| Model | Parameters | MAE C | RMSE C | Train seconds |
|---|---:|---:|---:|---:|
| Attention | 3168 | 1.55932750 | 2.05720942 | 19.032 |
| Daily | 0 | 2.05315123 | 2.71391287 | — |

Original MLP32 1.39610378; MLP64 1.38652827. Attention underperforms these in this controlled initial screen. Compact single-head encoder, no normalization, fixed training budget and seed; not a canonical transformer or family-level conclusion. All 62 records preserved. No new holdout access.
