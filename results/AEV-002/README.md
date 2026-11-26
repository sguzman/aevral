# AEV-002 — History-length validation

Code d029af423adab2c27b12bd2615b755f5a94feda2; [CI](https://github.com/sguzman/aevral/actions/runs/45865113357).
All 23 AEV-001 annual hashes verified. Matched 126093 train / 31045 validation origins; no test access. Full metrics: metrics.txt.

| Input hours | Parameters | MAE C | RMSE C | Skill vs daily |
|---:|---:|---:|---:|---:|
| 6 | 1016 | 2.12570156 | 2.85426634 | -0.03666353 |
| 12 | 1208 | 1.77374181 | 2.34466403 | 0.13498043 |
| 24 | 1592 | 1.65361398 | 2.20138649 | 0.19356445 |
| 48 | 2360 | 1.59926370 | 2.12970396 | 0.22007003 |
| 72 | 3128 | 1.59702117 | 2.12970750 | 0.22116366 |

Matched baselines MAE: persistence 2.99577189, daily 2.05052217, climatology 2.58881076. Select 72 hours by predeclared MAE criterion, but 48 hours is nearly tied and slightly better on RMSE. Single seed, correlated cases, unequal capacities and matched complete-history selection limit inference. No statistical significance claim. Calendar inputs next.
