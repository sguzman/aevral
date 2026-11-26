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
