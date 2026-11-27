# C007 final report — kernel, local and instance-based methods

Complete: 41/41 screens, 82/82 additional annual outcomes, 123 unique measured results, zero blocks. All 82 annual jobs passed, 120 Rust tests, 22 finalists plus reused C003 MLP64 seed42. 30,138 matched annual origins and 723,312 targets per seed. No new data or 2023+ tuning.

## Pooled ranking

Weight annual MAE by origin counts; pooled RMSE is the square root of weighted annual squared RMSE. For RFF pool within each seed then report arithmetic mean and sample SD (n−1) across seeds 42,123,2026. Seed replicates are not additional independent weather observations. Computed from rounded archived scores, not independent rescoring of raw prediction artifacts. The legacy C003 split is excluded.

| Rank | Family | Model | MAE °C | Seed SD MAE | RMSE °C | Seed SD RMSE |
|---:|---|---|---:|---:|---:|---:|
| 1 | incumbent | mlp64 | 1.34781461 | — | 1.80956416 | — |
| 2 | K04 | local-k128-l01 | 1.40971059 | — | 1.88793639 | — |
| 3 | K09 | fitc-s1-m256 | 1.43321565 | — | 1.93425430 | — |
| 4 | K04 | local-k64-l01 | 1.44627212 | — | 1.93873834 | — |
| 5 | K06 | rbf-krr-s1-l1 | 1.45057462 | — | 1.95225701 | — |
| 6 | K07 | svr-s1-c10 | 1.46908545 | — | 1.98386289 | — |
| 7 | K07 | svr-s1-c1 | 1.48750585 | — | 2.00941449 | — |
| 8 | K09 | fitc-s1-m128 | 1.49332771 | — | 2.01122652 | — |
| 9 | K08 | gp-s1 | 1.50047263 | — | 2.02191138 | — |
| 10 | K06 | rbf-krr-s1-l01 | 1.50404306 | — | 2.02740322 | — |
| 11 | K05 | linear-krr-l001 | 1.60664158 | — | 2.11082009 | — |
| 12 | K08 | gp-s05 | 1.67378290 | — | 2.30933333 | — |
| 13 | K02 | analog-k16-inverse | 1.68993041 | — | 2.21972778 | — |
| 14 | K02 | analog-k16-gaussian | 1.69192305 | — | 2.22264659 | — |
| 15 | K01 | knn-k16 | 1.69678658 | — | 2.22903864 | — |
| 16 | K01 | knn-k32 | 1.71827619 | — | 2.25582656 | — |
| 17 | K03 | season-k16-d30 | 1.73668371 | — | 2.29228669 | — |
| 18 | K03 | season-k8-d30 | 1.74007869 | — | 2.29473057 | — |
| 19 | K05 | linear-krr-l01 | 1.79876084 | — | 2.34405717 | — |
| 20 | K11 | nystrom-s1-m256 | 1.99714006 | — | 2.69145034 | — |
| 21 | K11 | nystrom-s1-m128 | 2.00494617 | — | 2.70336823 | — |
| 22 | K10 | rff-s1-m256 | 2.02336176 | 0.01985428 | 2.72397384 | 0.02369385 |
| 23 | K10 | rff-s1-m128 | 2.04142740 | 0.03859982 | 2.75145502 | 0.05449416 |

Best C007 model local-k128-l01: 1.40971059 MAE, 4.59% above MLP64. Local ridge and FITC are promising for efficient non-neural forecasting, not accuracy winners. C006 LightGBM-128 (1.40602200) remains slightly ahead of the best C007 model. No significance test for the difference. C007 recipes use 1024, 2048 or 8192 deterministic training origins; incumbent uses full causal training cohorts. Exact GP vs FITC comparison confounds training budgets. Approximation-error checks are training-pair diagnostics, not predictive guarantees. GP variance is not calibrated.

## Resource and reproducibility notes

Runner: GitHub ubuntu-latest; max six parallel, 30min/job, model workspace guards unchanged. Per-job wall times, RSS, preflight and algorithm diagnostics retained in archived logs; examples from 2022: local K128 6.17s / 368260 KiB process RSS; FITC256 1.12s / 368488 KiB; SVR C10 36.05s / 368480 KiB (elapsed includes command overhead, not whole CI job). No full-archive dense kernel matrix. Official LIBSVM 3.33 through locked Nix shell.

[Registry](registry.csv), [pooled scores](pooled.csv), [per-seed scores](pooled-seeds.csv), [frozen finalists](../../campaigns/007-kernels-neighbors/FINALISTS.md), [annual CI](https://github.com/sguzman/aevral/actions/runs/46029710841), [closure audit](closure-audit.md). Raw prediction artifacts have 30-day retention; aggregates and diagnostics are durable. C008 next.
