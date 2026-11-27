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
