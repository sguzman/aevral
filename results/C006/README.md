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

## T05/T09 histogram and median boosting
All six deterministic configs measured on 7455 matched origins, 24576 training rows. [CI](https://github.com/sguzman/aevral/actions/runs/45916492859), code 70e8d8ea2e0c439fb0f94fdbbfad14cb3235692d. 92 tests passed. All stages verified, finite normalized training objectives reduced in every configuration, no guard blocks.

| Model | MAE | RMSE | Fit seconds | Initial training loss | Final training loss |
|---|---:|---:|---:|---:|---:|
| hist-64-d5 | 1.40667273 | 1.85917069 | 6.765200 | 0.9981019256 | 0.0832750169 |
| hist-32-d5 | 1.49068693 | 1.95185005 | 3.341737 | 0.9981019256 | 0.0961489294 |
| hist-64-d3 | 1.50771902 | 1.96773303 | 4.165278 | 0.9981019256 | 0.1028620487 |
| median-boost-64 | 1.50829159 | 1.98343012 | 4.959911 | 0.3961299292 | 0.1227924696 |
| hist-32-d3 | 1.67603074 | 2.17361435 | 2.380178 | 0.9981019256 | 0.1277669879 |
| median-boost-32 | 1.74343882 | 2.31980780 | 2.392159 | 0.3961299292 | 0.1414954522 |

Best hist-64-d5 MAE1.40667273 RMSE1.85917069; best median-boost-64 MAE1.50829159. 22/28 screens measured. MLP64 remains better; different training sample budget. T06–T08 external backends pending.
