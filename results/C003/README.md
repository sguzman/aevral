# C003 benchmark results — complete

[Five successful benchmark jobs](https://github.com/sguzman/aevral/actions/runs/45892792311), code 20d721292f9d6c08148a6dd8fc632e74d44f2909. All 27 hashes verified, 33 tests passed per job. Legacy exact: 121849/30207, daily 2.05315123, MLP64 1.35014439. Annual folds exclude target windows crossing year boundaries; development, not independent.

| Fold | Training origins | Matched evaluation | Possible | Temperature complete | MLP64 MAE | Daily MAE |
|---|---:|---:|---:|---:|---:|---:|
| 2019 | 121849 | 7455 | 8737 | 7673 | 1.32617622 | 1.88752763 |
| 2020 | 129327 | 7543 | 8761 | 7719 | 1.30737647 | 2.07087759 |
| 2021 | 136893 | 7501 | 8737 | 7714 | 1.39099523 | 2.14898785 |
| 2022 | 144417 | 7639 | 8737 | 7870 | 1.36646120 | 2.10544725 |

Pooled 30138 origins, 723312 targets. Weighted from rounded logged fold metrics (MAE weighted by origin count, RMSE weighted squared):

| Model | MAE °C | RMSE °C |
|---|---:|---:|
| mlp64 | 1.34781461 | 1.80956416 |
| daily | 2.05372678 | 2.71500203 |
| persistence | 2.98494503 | 4.03723660 |
| climatology | 2.59725014 | 3.43176569 |

Raw logged aggregate, lead, month, training, memory and normalization records in individual fold files. [CI predictions artifacts](https://github.com/sguzman/aevral/actions/runs/45892792311) expire after 30 days. [Export audit](export-audit.txt): all five jobs passed, 5,793,120 prediction rows verified (annual plus overlapping legacy reproduction), exact target alignment, independent MAE/RMSE recomputation and 35 implementation tests. Artifact retention is finite; aggregate records are durable.
