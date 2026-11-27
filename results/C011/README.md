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

## R05–R09

10/10 measured, zero blocked. 185 tests per job, all ten artifact names reconciled, independent CI aggregate and per-lead rescoring passed; origin hashes identical to R00–R04. [CI](https://github.com/sguzman/aevral/actions/runs/46071772050).

| Recipe | Track | Matched MAE °C | RMSE °C | All optional observed (eval) |
|---|---|---:|---:|---:|
| pressure-tendency-long | B-expanded | 1.49161035 | 1.94371975 | 7455 |
| dew-depression-add | A-derived | 1.49766716 | 1.94963179 | n/a |
| weather-history-long | B-expanded | 1.49821302 | 1.93968125 | 6842 |
| weather-history-short | B-expanded | 1.49938380 | 1.95334423 | 6887 |
| wind-speed | A-derived | 1.50584971 | 1.95563110 | n/a |
| pressure-tendency-short | B-expanded | 1.52076605 | 1.97048449 | 7455 |
| dew-depression-replace | A-derived | 1.52916880 | 1.99072274 | n/a |
| harmonics-2-4 | A-derived | 1.53064439 | 1.99371573 | n/a |
| harmonics-2-3 | A-derived | 1.54194257 | 1.98884336 | n/a |
| wind-unit | A-derived | 1.54274905 | 2.00430055 | n/a |

Best overall screen remains suffix24 (1.43152429°C). All new recipes have 7455 matched and natural origins by design. Raw payloads not separately downloaded. Next R10–R14.

## R10–R14

10/10 measured, zero blocked. 191 tests per job; all ten artifact names reconciled, independent CI aggregate and per-lead rescoring passed; origin hashes identical to prior screens. Climatology counts and feature selection dimensions verified. [CI](https://github.com/sguzman/aevral/actions/runs/46072704541).

| Recipe | Track | Matched MAE °C | RMSE °C | All optional observed (eval) |
|---|---|---:|---:|---:|
| interactions-local | A-derived | 1.51971001 | 1.97447201 | n/a |
| spectral-5 | A-derived | 1.52211204 | 1.98607699 | n/a |
| spectral-3 | A-derived | 1.52606380 | 1.97895526 | n/a |
| anomaly-month | A-derived | 1.54857460 | 2.00287991 | n/a |
| anomaly-month-hour | A-derived | 1.55396889 | 2.01407648 | n/a |
| history96 | B-expanded | 1.55773682 | 2.02633246 | 7230 |
| interactions-calendar | A-derived | 1.56619058 | 2.04646281 | n/a |
| select48-leads | A-derived | 1.61614581 | 2.10049705 | n/a |
| history168 | B-expanded | 1.63721042 | 2.10089614 | 6626 |
| select32-mean | A-derived | 1.64832539 | 2.15528575 | n/a |

All 29 screens measured. Best screen suffix24 (1.43152429°C). All recipes advance under frozen manifest. Raw payloads not separately downloaded. No annual conclusions.
