# AEV-010 dense screen
[CI](https://github.com/sguzman/aevral/actions/runs/45876819315), code 40537c2e2792acaed5f1e031b5d553c724c1730a. 121849 training / 30207 validation origins; frozen 80 inputs. Original MLP reproduced exactly.
| Model | Parameters | MAE C | RMSE C | Train seconds |
|---|---:|---:|---:|---:|
| linear | 1944 | 1.56557755 | 2.07096501 | 4.060 |
| MLP16 | 1704 | 1.43482265 | 1.91319036 | 4.474 |
| MLP32 | 3384 | 1.39610378 | 1.86723672 | 8.334 |
| MLP64 | 6744 | 1.38652827 | 1.85783513 | 15.629 |
| MLP32x32 | 4440 | 1.39090582 | 1.86091240 | 11.750 |

Daily MAE 2.05315123. MLP64 leads this fixed-budget single-seed screen, not a significance or optimality claim. Linear SGD is not a least-squares optimum. All 206 metric records preserved. No 2026 holdout access.
