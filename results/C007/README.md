# C007 kernel and instance-based results
Frozen manifest: campaigns/007-kernels-neighbors/manifest.md. Full matched 2019 evaluation (7455 origins), 8192 evenly spaced train rows for K01–K03, deterministic. MLP64 incumbent 2019 MAE1.32617622 (full training). [CI](https://github.com/sguzman/aevral/actions/runs/45919448602), code 1aa79ac6e10ec5e2d6fbba190f4963cdb8fbe29c. 100 tests passed in completed jobs. 12/12 initial configurations archived; 12 measured.

| Model | MAE °C | RMSE °C | Prediction seconds | Min candidates | Mean candidates | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| analog-k16-inverse | 1.58604072 | 2.05652652 | 4.384990 | 8192 | 8192.00000000 | 0 |
| analog-k16-gaussian | 1.58769915 | 2.05840897 | 4.962320 | 8192 | 8192.00000000 | 0 |
| knn-k16 | 1.59227390 | 2.06374566 | 5.055882 | 8192 | 8192.00000000 | 0 |
| knn-k32 | 1.60847813 | 2.08038736 | 4.434725 | 8192 | 8192.00000000 | 0 |
| knn-k8 | 1.61394787 | 2.09788138 | 5.578213 | 8192 | 8192.00000000 | 0 |
| analog-k64-inverse | 1.64344166 | 2.11974886 | 5.353263 | 8192 | 8192.00000000 | 0 |
| analog-k64-gaussian | 1.64695321 | 2.12412833 | 5.527084 | 8192 | 8192.00000000 | 0 |
| season-k16-d30 | 1.65237903 | 2.17127893 | 0.516218 | 233 | 281.15613682 | 0 |
| knn-k64 | 1.65412064 | 2.13295204 | 5.355951 | 8192 | 8192.00000000 | 0 |
| season-k8-d30 | 1.65894053 | 2.17703494 | 0.489748 | 233 | 281.15613682 | 0 |
| season-k8-d15 | 1.72534653 | 2.27915218 | 0.344558 | 107 | 141.02602280 | 0 |
| season-k16-d15 | 1.75370376 | 2.32756712 | 0.419426 | 107 | 141.02602280 | 0 |

## K04 local linear ridge
[CI](https://github.com/sguzman/aevral/actions/runs/45919760217), code 017082c9c28f6650f9c1e1bfa38badd580509707, 105 tests passed. 8192 reference origins, full 7455 matched evaluation origins. Exact nearest neighbors, 12 slope coordinates, 24 outputs and unpenalized intercept. 4/4 measured.

| Model | MAE °C | RMSE °C | Max relative solve residual | Prediction seconds |
|---|---:|---:|---:|---:|
| local-k128-l01 | 1.37195009 | 1.82169663 | 1.698765516565e-16 | 6.399997 |
| local-k64-l01 | 1.41022498 | 1.88008412 | 1.755252078771e-16 | 5.668736 |
| local-k128-l1 | 1.44298756 | 1.88404816 | 1.616555399026e-16 | 5.725801 |
| local-k64-l1 | 1.44765587 | 1.89240256 | 1.678731308432e-16 | 5.980704 |

## K05–K06 exact kernel ridge
[CI](https://github.com/sguzman/aevral/actions/runs/45920051360), code 266013f3b5ffa2c7495f383f262c16f9c1044b68, 109 tests passed. Linear: 8192 rows, centered primal, mean-loss lambda; RBF: 2048 rows, exact dense subset, diagonal lambda. Full 7455 matched evaluation origins. 7/7 measured.

| Model | MAE °C | RMSE °C | Max relative solve residual | Fit seconds | Prediction seconds |
|---|---:|---:|---:|---:|---:|
| rbf-krr-s1-l1 | 1.39084668 | 1.84799529 | 6.248636469077e-13 | 1.138147 | 0.561371 |
| rbf-krr-s1-l01 | 1.46211482 | 1.95336877 | 5.649738436462e-12 | 1.822472 | 1.066933 |
| rbf-krr-s05-l1 | 1.48888541 | 1.99187113 | 1.552389932743e-13 | 2.109219 | 1.270754 |
| linear-krr-l001 | 1.51224494 | 1.97731958 | 8.926162933919e-17 | 0.045736 | 0.009457 |
| rbf-krr-s05-l01 | 1.67139318 | 2.22782417 | 6.117090814164e-13 | 1.192567 | 0.563556 |
| linear-krr-l01 | 1.68779093 | 2.19417262 | 6.394666979181e-17 | 0.044525 | 0.009520 |
| linear-krr-l1 | 2.24991204 | 2.94975527 | 1.018073504510e-16 | 0.039166 | 0.008301 |

