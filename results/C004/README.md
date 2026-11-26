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
