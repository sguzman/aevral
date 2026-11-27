# C008 screening results

D01 Fourier and D02 Haar: 8/8 measured, 125 tests passed, 7455 matched origins per configuration. Train budget 8192. [CI](https://github.com/sguzman/aevral/actions/runs/46031577648). C003 MLP64 2019 MAE 1.32617622; scores below from 2019 only.

| Family | Model | MAE °C | RMSE °C |
|---|---|---:|---:|
| D01 | fourier-k6-l001 | 1.47326621 | 1.94231164 |
| D02 | haar-m32-l001 | 1.47457748 | 1.94497482 |
| D02 | haar-m32-l01 | 1.47679777 | 1.94572315 |
| D02 | haar-m16-l001 | 1.47720748 | 1.94815570 |
| D01 | fourier-k6-l01 | 1.47935499 | 1.94611855 |
| D02 | haar-m16-l01 | 1.48368286 | 1.95265056 |
| D01 | fourier-k3-l001 | 1.52516603 | 1.99316512 |
| D01 | fourier-k3-l01 | 1.52535043 | 1.99146287 |

Full annual promotion follows completion of all nine families; no retuning.

## D03–D05 dynamical systems

[CI](https://github.com/sguzman/aevral/actions/runs/46032001261), 131 numerical tests, 8192 training windows, 7455 matched evaluation origins per measured configuration. Pooled training-window SSA; truncated delay DMD; finite polynomial EDMD. No future observations or physical-system identification claims.

| Family | Model | MAE °C | RMSE °C | Status |
|---|---|---:|---:|---|
| D04 | dmd-l24-r8 | 1.63639364 | 2.15934098 | measured |
| D03 | ssa-l24-r8 | 1.73745553 | 2.27052059 | measured |
| D03 | ssa-l36-r8 | 1.79963437 | 2.35929816 | measured |
| D03 | ssa-l24-r4 | 1.81060547 | 2.35703302 | measured |
| D04 | dmd-l24-r4 | 1.81813557 | 2.34990010 | measured |
| D03 | ssa-l36-r4 | 1.88232594 | 2.44460758 | measured |
| D05 | edmd-l12-l001 | 2.05963056 | 2.71789011 | measured |
| D05 | edmd-l12-l01 | 2.16448064 | 2.85948682 | measured |
| D04 | dmd-l12-r8 | 2.18391565 | 2.89165835 | measured |
| D04 | dmd-l12-r4 | 2.76016438 | 3.66100304 | measured |
| D05 | edmd-l6-l001 | 3.06486570 | 4.12766962 | measured |
| D05 | edmd-l6-l01 | 3.18803616 | 4.27526135 | measured |

## D06–D07 random-feature learners

[CI](https://github.com/sguzman/aevral/actions/runs/46032441990), 137 numerical tests, 8192 training windows, 7455 matched evaluation origins per measured configuration. Seed42 screening only; fixed hidden weights and trained ridge readout. Both families are neural/hybrid; three-seed finalists will follow promotion.

| Family | Model | MAE °C | RMSE °C | Status |
|---|---|---:|---:|---|
| D07 | elm-m128-l001 | 1.48624269 | 1.94727113 | measured |
| D07 | elm-m128-l01 | 1.49773178 | 1.95946007 | measured |
| D06 | esn-m128-leak03 | 1.57284555 | 2.05669316 | measured |
| D06 | esn-m64-leak03 | 1.58162346 | 2.06914311 | measured |
| D07 | elm-m64-l001 | 1.60349399 | 2.08622951 | measured |
| D07 | elm-m64-l01 | 1.60754493 | 2.09246551 | measured |
| D06 | esn-m128-leak1 | 1.64314148 | 2.16326497 | measured |
| D06 | esn-m64-leak1 | 1.64990894 | 2.17280684 | measured |
