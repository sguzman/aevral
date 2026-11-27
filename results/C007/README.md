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


## K07 official LIBSVM epsilon-SVR
[CI](https://github.com/sguzman/aevral/actions/runs/45920736527), code d3ba83de6f6a27925948778e7ff680249284cb32, 112 tests passed. Pinned LIBSVM 3.33, 2048 train rows, 24 independent scalar fits, epsilon .05, RBF kernel, full 7455 matched evaluation origins. 4/4 measured. Sum of support vectors across leads is not a count of unique training origins.

| Model | MAE °C | RMSE °C | Total support vectors across leads | Fit and prediction seconds |
|---|---:|---:|---:|---:|
| svr-s1-c1 | 1.40649144 | 1.87076562 | 41329 | 20.537950 |
| svr-s1-c10 | 1.41467660 | 1.89047014 | 41160 | 18.905404 |
| svr-s05-c1 | 1.45846807 | 1.95465293 | 41653 | 26.630543 |
| svr-s05-c10 | 1.62612259 | 2.16276423 | 42122 | 22.855389 |


## K08–K09 exact and FITC Gaussian processes
[CI](https://github.com/sguzman/aevral/actions/runs/45921039141), code d48e02b0bebe02729d5e64e529859e8cca6a6a7e, 116 tests passed. Exact GP 1024 rows; streaming FITC 8192 rows, 128/256 inducing points. Shared covariance, 24 scalar outputs, fixed noise .05 and jitter 1e-8. Full 7455 matched evaluation origins. 6/6 measured. Latent variance diagnostics first32 evaluation inputs, normalized units, not calibrated uncertainty.

| Model | MAE °C | RMSE °C | Mean latent variance (first32) | Fit seconds | Prediction seconds |
|---|---:|---:|---:|---:|---:|
| fitc-s1-m256 | 1.37723166 | 1.82825816 | 1.181206235502e-2 | 0.529687 | 0.135221 |
| fitc-s1-m128 | 1.41968820 | 1.87521576 | 1.969707387006e-2 | 0.205052 | 0.080702 |
| gp-s1 | 1.44283129 | 1.90324250 | 1.270628046215e-2 | 0.258039 | 0.548060 |
| fitc-s05-m256 | 1.53950769 | 2.07118549 | 1.370623360777e-1 | 0.528327 | 0.135327 |
| gp-s05 | 1.58191727 | 2.10234541 | 8.053827432377e-2 | 0.151878 | 0.273447 |
| fitc-s05-m128 | 1.63503792 | 2.20456565 | 1.872717991558e-1 | 0.126594 | 0.038258 |

## K10–K11 kernel feature approximation
[CI](https://github.com/sguzman/aevral/actions/runs/45921413274), code f5b6ad4b79c7473eaec3a5de49a4e49d78b521e8, 120 tests passed. 8192 training rows, 128/256 features or landmarks, ridge .01; RFF seed42, deterministic Nystrom. All 8 jobs passed and 7455 matched origins per model; 24 leads and 12 months validated. Kernel error diagnostics are deterministic training-pair checks, not generalization guarantees.

| Model | MAE °C | RMSE °C | Fit seconds | Prediction seconds |
|---|---:|---:|---:|---:|
| nystrom-s1-m256 | 1.84759723 | 2.41470335 | 0.756765 | 0.316629 |
| nystrom-s1-m128 | 1.85517621 | 2.42631308 | 0.267587 | 0.130463 |
| rff-s1-m256 | 1.88266398 | 2.45313193 | 0.613885 | 0.172226 |
| rff-s1-m128 | 1.92873945 | 2.51186785 | 0.139266 | 0.046426 |
| nystrom-s05-m256 | 1.99424913 | 2.72614282 | 0.748301 | 0.322254 |
| rff-s05-m256 | 2.00721237 | 2.69311848 | 0.424511 | 0.118175 |
| nystrom-s05-m128 | 2.03367341 | 2.78397943 | 0.218043 | 0.093396 |
| rff-s05-m128 | 2.21448817 | 2.98282368 | 0.162789 | 0.062178 |

All 41 screens measured; no blocked configurations. Best local-k128-l01 MAE 1.37195009. [Frozen finalists](../../campaigns/007-kernels-neighbors/FINALISTS.md): 22 configurations, 82 additional annual outcomes pending.
