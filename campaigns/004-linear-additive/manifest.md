# C004 finite experiment manifest v1 — linear and additive models
Predeclared before any C004 fit. Fixed 80-input Track A, 2019 development screen, C003 data and scorer. Target normalized using training-history temperature mean/SD; score Celsius MAE/RMSE. Standardize additional basis features from train only; intercept unpenalized. Deterministic algorithms, one run per fold. Reference incumbent is C003, no refit needed for identical data/recipe. No holdout parsing. No external data.

## Candidate inventory (screening)
| ID | Method | Predeclared settings | Count |
|---|---|---|---:|
| L01 | OLS multi-output | stable QR (streamed/block or direct), rank-aware solve; fail rather than silent ridge | 1 |
| L02 | Ridge | lambda = 1e-6,1e-4,1e-3,1e-2,0.1,1 | 6 |
| L03 | Lasso | lambda = 1e-4,1e-3,1e-2,0.1 | 4 |
| L04 | Elastic Net | lambda = 1e-3,1e-2,0.1; L1 ratio .25,.75 | 6 |
| L05 | Huber | delta=.25,.5,1,2 normalized units | 4 |
| L06 | Median regression | pinball tau=.5, L2 = 0,1e-4,1e-2 | 3 |
| L07 | Principal-component regression | 8,16,32,64 components | 4 |
| L08 | Partial least squares | 4,8,16,32 latent components | 4 |
| L09 | Reduced-rank regression | rank 2,4,8,16 | 4 |
| L10 | Polynomial interactions | degree-2 all 80 inputs or predefined recent-weather interactions; ridge .001,.01,.1 | 6 |
| L11 | GAM | additive piecewise linear spline bases; 3,7 knots per input; ridge .001,.01,.1 | 6 |
| L12 | MARS | forward hinge selection and backward pruning; 16,32,64 max terms, degree 1 or 2 | 6 |

Total 54 screening configurations, within 96 limit. Lambda uses normalized objective: mean squared residual (factor 1/2) plus lambda penalty; ridge penalty lambda/2 * squared norm, lasso lambda * L1, elastic net convex combination. Huber and median losses defined per output; report solver tolerances, convergence and actual wall time. For polynomial terms use resource-preflight; if full basis exceeds workspace/time budget document block and retain faithful restricted interactions. MARS must perform real adaptive hinge selection/pruning, not merely static splines; if infeasible document. Use exact algorithms or explicit approximate variant labels. Do not substitute ridge for OLS, PCR for PLS or static GAM for MARS. Set solver-specific finite tolerance/iterations before each batch. Implementation and synthetic numerical correctness gate before CI fit.

## Execution order
1. L01/L02 stable least squares and ridge (initial batch). Shared predictor trait, intercept, diagnostics, QR/rank test, known coefficients, ridge shrinkage, prediction parity. Confirm exact frozen 2019 count 7455 and baseline.
2. L03/L04 sparse regularization.
3. L05/L06 robust and absolute-error objectives.
4. L07–L09 latent and reduced-rank regression.
5. L10–L12 interactions and additive/adaptive spline models.
6. Promote best two configurations per method (one where only one exists) by 2019 MAE, tie-break RMSE then fit+inference cost. Evaluate on all four annual folds (reuse 2019 measured run); report pooled ranking, fit time, RSS, effective degrees/parameter count, lead and month metrics, failed/skipped configurations. If infeasible under budget, explicit reason; no invented results. No paid compute. CI <=6 parallel, <=30 min per job, <=2 GB workspace target. Persist configs and outcomes in results/C004; no tuning on 2023–2026.
