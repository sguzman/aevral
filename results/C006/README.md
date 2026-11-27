# C006 results — trees and boosting

T01 complete: six deterministic configurations, all measured on the matched 7455-origin 2019 screen, 24576 training rows. [CI](https://github.com/sguzman/aevral/actions/runs/45914941755), code b7a2394213088bd2e6eac28d39a43e73cb4d61db. 81 tests passed. All fit below guard, peak RSS <315000 kB. No new data.

| Model | MAE | RMSE | Fit seconds |
|---|---:|---:|---:|
| cart-lead-d8-l32 | 1.61833904 | 2.14853219 | 18.287183 |
| cart-shared-d8-l32 | 1.74356087 | 2.26458600 | 1.025419 |
| cart-shared-d8-l128 | 1.79325173 | 2.32980566 | 0.743771 |
| cart-lead-d4-l32 | 1.83625818 | 2.38399158 | 10.903146 |
| cart-shared-d4-l32 | 2.30868439 | 2.95386649 | 0.666427 |
| cart-shared-d4-l128 | 2.30868439 | 2.95386649 | 0.658907 |

Independent per-lead depth8 beats corresponding shared tree on 2019 but costs more fit time; MLP64 incumbent MAE 1.32617622 remains better. Compare shared and per-lead subfamilies separately for promotion. Annual selection not yet launched. T02–T09 pending.

## T02–T04 randomized ensembles
All ten measured on matched 7455 origins, 24576 training rows, seed42. [CI](https://github.com/sguzman/aevral/actions/runs/45915256233), code 28b7f3e510da416885d33dc83dbc2ffe6beb6425. 86 tests passed; no guard blocks.

| Model | MAE | RMSE | Fit seconds |
|---|---:|---:|---:|
| bag-32 | 1.54528611 | 2.00965579 | 14.768016 |
| bag-16 | 1.55441162 | 2.02334710 | 10.544321 |
| rf-32-27 | 1.55554826 | 2.02457661 | 7.356862 |
| rf-16-27 | 1.56552635 | 2.03734103 | 2.560282 |
| rf-32-9 | 1.62549994 | 2.10692104 | 2.713217 |
| rf-16-9 | 1.63939819 | 2.12581505 | 1.767559 |
| extra-32-27 | 1.65498500 | 2.14123468 | 1.904453 |
| extra-16-27 | 1.66425134 | 2.15079650 | 0.799992 |
| extra-32-9 | 1.79319612 | 2.30945923 | 0.979526 |
| extra-16-9 | 1.79775366 | 2.31958249 | 0.351265 |

Best bag-32 1.54528611, rf-32-27 1.55554826, extra-32-27 1.65498500. MLP64 remains better. Screening seed only; annual replicates required. 16/28 screens measured.
