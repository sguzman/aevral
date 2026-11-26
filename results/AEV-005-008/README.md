# AEV-005–008 — Meteorological feature selection

[Successful CI](https://github.com/sguzman/aevral/actions/runs/45865789255), code 145c7d16f807c9e42208b96c4074df3bdc76aead. Frozen 2000–2022 snapshot; 121849 train and 30207 validation origins shared by all candidates. No test access. Daily persistence MAE 2.05315123 C.

| Features beyond 72h temperature + both calendar cycles | MAE C | RMSE C | Skill vs daily |
|---|---:|---:|---:|
| None | 1.55249239 | 2.06851652 | 0.24384899 |
| Dew | 1.51533455 | 2.03037428 | 0.26194694 |
| Pressure | 1.49441824 | 1.99583684 | 0.27213436 |
| Dew + pressure | 1.48072892 | 1.98377355 | 0.27880182 |
| Wind | 1.45958670 | 1.95237230 | 0.28909928 |
| Dew + wind | 1.44731947 | 1.93825409 | 0.29507410 |
| Pressure + wind | 1.41205144 | 1.88417785 | 0.31225162 |
| All | 1.39610378 | 1.86723672 | 0.32001903 |

All features selected by predeclared MAE criterion. Wind has the largest individual improvement. Full model reduces MAE about 10.07% versus the matched reference. Exact scale, epoch losses and lead scores are preserved in metrics.txt. Single seed, overlapping windows, unequal parameter counts, and complete-case selection limit conclusions; no causal or significance claim. Validation only.
