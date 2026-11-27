# C009 screen results

Frozen 2019 cohort: 7,455 origins, seed42, 8,192 training origins, 12 epochs Adam .001. C003 full-training MLP64 2019 MAE 1.32617622 (different training budget). Six of 42 screened, no blocked. Rust and historical architecture regression CI passed. No annual ranking yet.

| Recipe | MAE °C | RMSE °C | Parameters | Fit seconds | Predict seconds | Clipped updates |
|---|---:|---:|---:|---:|---:|---:|
| mlp-w128 | 1.52309663 | 1.97813457 | 13464 | 12.143077 | 0.110769 | 0 |
| resmlp-w64-b2 | 1.52771089 | 1.98283934 | 23384 | 9.932843 | 0.070316 | 817 |
| resmlp-w64-b4 | 1.53330066 | 2.00639008 | 40024 | 27.104756 | 0.218524 | 1249 |
| mlp-w256 | 1.53909188 | 1.99918547 | 26904 | 13.398324 | 0.128007 | 0 |
| mlp-128x64 | 1.59058790 | 2.06952853 | 20184 | 19.142084 | 0.166310 | 0 |
| resmlp-w128-b2 | 1.62040418 | 2.09164529 | 79512 | 34.728919 | 0.243631 | 1130 |

[CI](https://github.com/sguzman/aevral/actions/runs/46035179411) · [Registry](registry.csv).
