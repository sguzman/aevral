# C004 — Linear and additive model screening (in progress)

## L01/L02 OLS and ridge
[CI passed](https://github.com/sguzman/aevral/actions/runs/45896077849), code 523c83e41c686c3927bd8d612bd140f7562393ef, 38 tests passed, 27 archive hashes verified. 121849 training / 7455 evaluation origins, daily baseline 1.88752763 reproduced. All seven measured; full QR rank 81, 1944 coefficients per model. Shared QR factor time 0.438368 s; solve times per config in [logged records](L01-L02-2019.txt). Process peak RSS 314128 kB (includes loading). Separate CI run incumbent MLP64 1.32617622°C MAE, fit 73.016 s; timing not controlled hardware comparison.

| Configuration | MAE °C | RMSE °C |
|---|---:|---:|
| ridge-1e-4 | 1.46767092 | 1.93423733 |
| ridge-1e-6 | 1.46768776 | 1.93427520 |
| ols | 1.46768796 | 1.93427561 |
| ridge-1e-3 | 1.46772529 | 1.93409839 |
| ridge-1e-2 | 1.47327803 | 1.93811186 |
| ridge-0.1 | 1.52106684 | 1.98584682 |
| ridge-1 | 1.71675828 | 2.23311753 |

Promote OLS and ridge-1e-4, ridge-1e-6 by manifest (not yet measured on other years). No selection on exposed historical test periods. Next L03/L04. 7/54 screens measured.

## L03/L04 Lasso and Elastic Net
[CI passed](https://github.com/sguzman/aevral/actions/runs/45896275859), code 3f72dd58c7eb62f9dd8635f5f5dff25099e21604. 41 tests; same frozen cohort and baseline; all ten configurations reached KKT tolerance 1e-7. Gram time 0.345439 s; process peak RSS 314228 kB. Shared Gram plus individual solve time is standalone fit cost. Nonzero slopes out of 1920, excluding intercepts.

| Configuration | MAE °C | RMSE °C | Nonzero slopes | Max sweeps |
|---|---:|---:|---:|---:|
| lasso-1e-4 | 1.46776943 | 1.93442307 | 1394 | 1369 |
| enet-1e-3-a25 | 1.46814867 | 1.93471868 | 1214 | 1192 |
| enet-1e-3-a75 | 1.46977817 | 1.93663733 | 991 | 1220 |
| lasso-1e-3 | 1.47069685 | 1.93767182 | 926 | 1245 |
| enet-1e-2-a25 | 1.48065804 | 1.94721516 | 897 | 773 |
| enet-1e-2-a75 | 1.49592709 | 1.96778649 | 524 | 895 |
| lasso-1e-2 | 1.50531026 | 1.98007670 | 441 | 1111 |
| enet-0.1-a25 | 1.61343128 | 2.10861726 | 517 | 306 |
| enet-0.1-a75 | 1.74021823 | 2.27241500 | 245 | 392 |
| lasso-0.1 | 1.78597604 | 2.32658327 | 141 | 1028 |

Lasso 1e-4 removes 526/1920 slopes with negligible accuracy change relative to OLS. Sparse regularization does not improve the best ridge result. Promote lasso-1e-4, lasso-1e-3, enet-1e-3-a25 and enet-1e-3-a75 after screening. [Full records](L03-L04-2019.txt). 17/54 screens measured.

## L05/L06 Huber and median regression
[CI passed](https://github.com/sguzman/aevral/actions/runs/45896640395), code 8546ecef5d3f72b4b365f2331383b9c2648f4052. All seven jobs passed 45 tests and frozen cohort/hash checks; all seven converged under the declared primal and stationarity tolerances. Full records in L05-L06-*-2019.txt.

| Configuration | MAE °C | RMSE °C | Iterations | Fit s |
|---|---:|---:|---:|---:|
| huber-0.25 | 1.46480324 | 1.93582492 | 20 | 3.258994 |
| huber-0.5 | 1.46516741 | 1.93388172 | 20 | 2.991646 |
| median-1e-4 | 1.46546481 | 1.93801795 | 340 | 50.944243 |
| median-0 | 1.46548444 | 1.93805607 | 340 | 49.204258 |
| huber-1 | 1.46710856 | 1.93375828 | 20 | 3.158634 |
| median-1e-2 | 1.46756977 | 1.93998691 | 330 | 59.592293 |
| huber-2 | 1.46769903 | 1.93429318 | 20 | 2.514277 |

Huber .25 is best non-neural model so far, but only marginally improves ridge. Promote huber-0.25, huber-0.5, median-1e-4 and median-0 by MAE. Median optimizes pinball loss (half absolute error) rather than squared error; no claim of independent test performance. 24/54 measured.

## L07–L09 PCR, PLS2, reduced-rank regression
[CI passed](https://github.com/sguzman/aevral/actions/runs/45897005852), code d61616dfe6197d8588d8426ea04264c239bd8b4e, 48 tests, frozen cohort and archive verified. Gram 0.367793 s; shared eigensystem 0.007953 s; process peak RSS 314448 kB. Standalone cost includes Gram, eigensystem for PCR/RRR, and per-config fit. Dense prediction implementation uses 1944 coefficients; latent rank is not a measured compression ratio.

| Configuration | MAE °C | RMSE °C | Latent rank | Additional fit s |
|---|---:|---:|---:|---:|
| rrr-16 | 1.46770656 | 1.93427248 | 16 | 0.000816 |
| pls-32 | 1.46782162 | 1.93435761 | 32 | 0.010656 |
| pcr-64 | 1.46788769 | 1.93442285 | 64 | 0.000207 |
| rrr-8 | 1.46814017 | 1.93478966 | 8 | 0.000720 |
| pcr-32 | 1.47943917 | 1.94460980 | 32 | 0.000102 |
| pls-16 | 1.48933475 | 1.95309473 | 16 | 0.004602 |
| rrr-4 | 1.49962065 | 1.96391690 | 4 | 0.000726 |
| pcr-16 | 1.57135276 | 2.03785400 | 16 | 0.000052 |
| pls-8 | 1.57848706 | 2.04098669 | 8 | 0.002297 |
| pcr-8 | 1.68897924 | 2.16728177 | 8 | 0.000022 |
| pls-4 | 1.73035750 | 2.23964347 | 4 | 0.001233 |
| rrr-2 | 1.96469435 | 2.55911565 | 2 | 0.000770 |

All 12 configurations measured; none beats Huber or ridge. RRR rank 8 nearly reproduces OLS. Promote pcr-64/pcr-32, pls-32/pls-16, rrr-16/rrr-8. [Full records](L07-L09-2019.txt). 36/54 screens measured.

## L10/L11 polynomial and additive basis regression (partial)
[CI](https://github.com/sguzman/aevral/actions/runs/45897341015), code 77848bc72e0a025eb2fd94c4a6ca8fe0d776eb82. Restricted polynomial and both GAM families passed all 51 tests and frozen archive/cohort checks. Full polynomial job still running, not counted here. All nine completed configurations met stationarity tolerance.

| Configuration | MAE °C | RMSE °C | Slopes per lead | Solve s |
|---|---:|---:|---:|---:|
| poly-recent-1e-3 | 1.33658679 | 1.76999918 | 608 | 0.037254 |
| poly-recent-1e-2 | 1.34343534 | 1.77492074 | 608 | 0.037093 |
| poly-recent-0.1 | 1.39834038 | 1.83418853 | 608 | 0.037144 |
| gam-7-1e-3 | 1.45231366 | 1.90959287 | 640 | 0.044112 |
| gam-3-1e-3 | 1.45551824 | 1.91301069 | 320 | 0.003375 |
| gam-7-1e-2 | 1.45583124 | 1.91323998 | 640 | 0.044340 |
| gam-3-1e-2 | 1.46066367 | 1.91816956 | 320 | 0.003911 |
| gam-7-0.1 | 1.47985154 | 1.93986904 | 640 | 0.045310 |
| gam-3-0.1 | 1.49430305 | 1.95472767 | 320 | 0.003895 |

Restricted quadratic ridge 1e-3 is best non-neural configuration so far (MAE 1.33658679; RMSE 1.76999918), both slightly worse than 2019 MLP64 (MAE 1.32617622, RMSE 1.76544057). Restricted quadratic training basis/Gram/solve ~7.24 s; separate CI timings not controlled. Full polynomial results pending before promotion. GAM candidates gam-7-1e-3 and gam-3-1e-3 (ranked by MAE) advance after screening. 45/54 screens measured.

## L12 bounded-grid MARS
[CI passed](https://github.com/sguzman/aevral/actions/runs/45897731820), code 613d8b1feda79032aa5466304ff9310afe170a3b. All six configurations measured; 53 tests and frozen cohort/hash checks passed. Forward selection uses a restricted sample/candidate grid; full training data used for coefficients and pruning. Full logs include selected hinges, forward and pruning paths.

| Configuration | MAE °C | RMSE °C | Retained terms (excluding intercept) | Fit s |
|---|---:|---:|---:|---:|
| mars-grid-2-64 | 1.43531178 | 1.88305991 | 63 | 0.738119 |
| mars-grid-1-64 | 1.47772393 | 1.94027705 | 54 | 0.363221 |
| mars-grid-1-32 | 1.49896128 | 1.96933186 | 32 | 0.172979 |
| mars-grid-2-32 | 1.52019852 | 1.99215867 | 32 | 0.392698 |
| mars-grid-1-16 | 1.55781347 | 2.04656335 | 16 | 0.062616 |
| mars-grid-2-16 | 1.58869578 | 2.08718744 | 16 | 0.100725 |

Promote mars-grid-2-64 and mars-grid-1-64 by MAE. 51/54 screens measured; full polynomial job still pending. No multi-year finalist runs yet.
