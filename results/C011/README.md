# C011 representation screens

R00–R04: 9/9 measured, zero blocked. All nine artifact names reconciled, 179 tests passed per job. All matched train/fit/evaluation origin-list hashes identical across recipes. Independent AWK aggregate and per-lead rescoring passed in CI (payloads not downloaded separately). Base80 exactly reproduces C010 joint MLP64 control. [CI](https://github.com/sguzman/aevral/actions/runs/46068434695). 2019 seed42 only; not annual rankings. Natural coverage separately scored for suffix recipes.

| Recipe | Track | Matched MAE °C | RMSE °C | Natural origins | Natural MAE °C |
|---|---|---:|---:|---:|---:|
| suffix24 | A-derived | 1.43152429 | 1.88168144 | 7948 | 1.42463709 |
| suffix48 | A-derived | 1.49396261 | 1.95490461 | 7691 | 1.48857094 |
| gradient-short | A-derived | 1.51419063 | 1.97310639 | 7455 | 1.51419063 |
| gradient-long | A-derived | 1.51537587 | 1.97013476 | 7455 | 1.51537587 |
| base80 | A-control | 1.52449139 | 1.98742079 | 7455 | 1.52449139 |
| sparse-lags-14d | B-expanded | 1.53109783 | 1.98776202 | 7455 | 1.53109783 |
| sparse-lags-28d | B-expanded | 1.55670503 | 2.00863852 | 7455 | 1.55670503 |
| rolling-3-12-48 | A-derived | 1.56744357 | 2.02615783 | 7455 | 1.56744357 |
| rolling-6-24-72 | A-derived | 1.59863373 | 2.04761670 | 7455 | 1.59863373 |

Suffix24 covers 493 additional origins; suffix48 covers 236. Sparse 14d/28d lag all-observed eval counts 7393/7370 of 7455, respectively. No unmatched-score comparison as paired. Next R05–R09.
