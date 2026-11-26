# C004 — linear and additive regression final accuracy report
54/54 screening configurations measured across twelve methods. Frozen top 23 finalists: 69 additional annual outcomes, 67 measured and two explicit preflight blocks. Total registry 123 unique model/fold outcomes, 121 measured, 2 blocked. 21/21 annual workflow jobs completed successfully (successful job != measured fit). 53 numerical tests on finalist runs. Incumbent reused C003, no new data or holdout tuning.

## Complete four-year ranking
Matched 30,138 origins, 723,312 lead targets per complete model. Pooled MAE = sum(n_y*MAE_y)/sum(n_y), pooled RMSE = sqrt(sum(n_y*RMSE_y²)/sum(n_y)), from rounded annual scores; no averaging RMSE directly. All deterministic C004 finalists one run per annual fold. MLP64 seed 42 incumbent. No claims of fresh holdout performance.

| Rank | Model | 2019 MAE | 2020 MAE | 2021 MAE | 2022 MAE | Pooled MAE | Pooled RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | MLP64 | 1.32617622 | 1.30737647 | 1.39099523 | 1.3664612 | 1.34781461 | 1.80956416 |
| 2 | mars-grid-2-64 | 1.43531178 | 1.45789590 | 1.54612242 | 1.49132800 | 1.48274195 | 1.98126367 |
| 3 | gam-7-1e-3 | 1.45231366 | 1.49597721 | 1.55259404 | 1.54739875 | 1.51230146 | 2.00543294 |
| 4 | gam-3-1e-3 | 1.45551824 | 1.49880766 | 1.55869295 | 1.55720233 | 1.51780540 | 2.01268222 |
| 5 | mars-grid-1-64 | 1.47772393 | 1.51874680 | 1.57455366 | 1.57126099 | 1.53579961 | 2.03316787 |
| 6 | median-0 | 1.46548444 | 1.52658737 | 1.59267851 | 1.59836453 | 1.54611531 | 2.05688391 |
| 7 | median-1e-4 | 1.46546481 | 1.52658023 | 1.59271682 | 1.59837567 | 1.54612102 | 2.05687762 |
| 8 | huber-0.25 | 1.46480324 | 1.52633734 | 1.59267192 | 1.59945837 | 1.54615984 | 2.05513524 |
| 9 | huber-0.5 | 1.46516741 | 1.52638445 | 1.59362073 | 1.60221924 | 1.54719765 | 2.05364557 |
| 10 | ols | 1.46768796 | 1.52642216 | 1.59683001 | 1.60693451 | 1.54982450 | 2.05377782 |
| 11 | ridge-1e-6 | 1.46768776 | 1.52642213 | 1.59683041 | 1.60693482 | 1.54982462 | 2.05377786 |
| 12 | rrr-16 | 1.46770656 | 1.52643201 | 1.59683373 | 1.60693926 | 1.54983369 | 2.05378248 |
| 13 | ridge-1e-4 | 1.46767092 | 1.52642202 | 1.59687109 | 1.60696817 | 1.54983900 | 2.05378430 |
| 14 | lasso-1e-4 | 1.46776943 | 1.52644650 | 1.59690298 | 1.60690204 | 1.54986067 | 2.05388354 |
| 15 | pcr-64 | 1.46788769 | 1.52653446 | 1.59699449 | 1.60704035 | 1.54996977 | 2.05389415 |
| 16 | pls-32 | 1.46782162 | 1.52653308 | 1.59695039 | 1.60715447 | 1.54997104 | 2.05391872 |
| 17 | rrr-8 | 1.46814017 | 1.52719839 | 1.59713603 | 1.60752215 | 1.55035575 | 2.05429255 |
| 18 | enet-1e-3-a25 | 1.46814867 | 1.52685695 | 1.59766602 | 1.60764557 | 1.55043558 | 2.05446961 |
| 19 | enet-1e-3-a75 | 1.46977817 | 1.52832501 | 1.59888030 | 1.60893357 | 1.55183478 | 2.05621757 |
| 20 | lasso-1e-3 | 1.47069685 | 1.52925869 | 1.59977175 | 1.60975426 | 1.55272560 | 2.05729569 |
| 21 | pcr-32 | 1.47943917 | 1.53872502 | 1.60914633 | 1.61824778 | 1.56174342 | 2.06433228 |
| 22 | pls-16 | 1.48933475 | 1.54705438 | 1.61844430 | 1.62730028 | 1.57088457 | 2.07216709 |

## Partial polynomial comparison (2019–2021 only)
Matched 22,499 origins; do not mix with four-year ranking. 2022 preflight projected 611.317939 s; conservative estimate 1402.636s > 1400s. Both configurations blocked before Gram fit, as required by L10-L11 predeclared guard. No score assigned; no budget relaxation or replacement configuration. All three L10 screening full configurations measured in 2019. The full polynomial model beat MLP64 in 2019 alone, but trails on 2020 and 2021. Its four-year ranking is unknown.

| Model | MAE | RMSE | Origins |
|---|---:|---:|---:|
| MLP64 | 1.34148361 | 1.80356329 | 22499 |
| poly-full-1e-3 | 1.34780595 | 1.81673061 | 22499 |
| poly-full-1e-2 | 1.35613527 | 1.82474374 | 22499 |

Best fully evaluated non-neural model: mars-grid-2-64. C004 winner among complete models remains MLP64. Lead, month, fit diagnostics, elapsed time and RSS retained in per-job logs; resource figures from different runners not controlled. Prediction artifacts expire after 30 days. Approximation scope for MARS documented in L12.md. Do not claim every conceivable regression variant exhausted.

CI [screen polynomial/GAM](https://github.com/sguzman/aevral/actions/runs/45897341015), [MARS](https://github.com/sguzman/aevral/actions/runs/45897731820), [finalists A](https://github.com/sguzman/aevral/actions/runs/45898046310), [finalists B](https://github.com/sguzman/aevral/actions/runs/45898314791). Raw registry and detailed logs in this directory.
