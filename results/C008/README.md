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
