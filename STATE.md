# Fixed-data temperature program — C003 complete, C004 active

C001/C002 historical reports immutable; both old holdouts exposed. C003 complete: 27 hashes verified, all five benchmark jobs passed 45892792311, legacy exact, 30138 matched annual origins (2019 7455, 2020 7543, 2021 7501, 2022 7639), pooled MLP64 MAE 1.34781461, daily 2.05372678. Export audit 45895728513 all five passed; 5793120 prediction rows verified, 35 tests, archived results/C003. Program campaigns/TEMPERATURE-PROGRAM.md. C004 manifest campaigns/004-linear-additive/manifest.md predeclares 54 screen configurations across 12 methods. Next implement L01/L02 stable least-squares and ridge first batch in Rust using src/features/tournament.rs API, test numerical correctness, add CLI/CI with frozen data, score 2019. No C004 results or code yet. Subsequent batches per manifest. No new data or local downloads. Keep logs filtered and state durable. No human action required.


## C004 phase 1
L01/L02 implemented streaming Givens QR, unpenalized intercept, explicit diagonal rank check and ridge augmentation, exact/shrinkage/rank tests, CLI and 2019 all-seven screen CI. Shared factorization; standalone fit cost factor+solve. Pending CI verification, archive actual results and compare with 2019 MLP64 MAE 1.32617622, then L03/L04. No results yet.


## C004 phase 2
L01/L02 CI 45896077849 passed, all seven scores and 168 leads archived in results/C004/L01-L02-2019.txt, registry and README. Best ridge-1e-4 1.46767092 MAE, OLS 1.46768796, incumbent 1.32617622. QR rank full; 38 tests. Promote OLS, ridge-1e-4, ridge-1e-6 after all screening. Next L03/L04 Lasso and Elastic Net (10 configs), implement centered Gram-based cyclic coordinate descent with unpenalized intercept, KKT convergence and predeclared finite cap; validate synthetic orthogonal shrinkage and objective. Same matched data and baseline. No further C004 runs yet. 7/54 measured.

## C004 phase 3
L03/L04 ten-config sparse screen implemented, centered Gram cyclic coordinate descent, KKT 1e-7/10000 sweeps, unpenalized intercept, tests analytic shrinkage/direct objective/exact zeros, CLI and CI. Await verify actual CI and archive scores/blocked configs, then L05/L06 robust/median objectives. 7/54 measured before this batch, no sparse scores yet.

## C004 phase 4
L03/L04 passed CI 45896275859, all ten converged KKT <=1e-7, 41 tests, archived 388 records in results/C004/L03-L04-2019.txt, README and registry. Best lasso-1e-4 MAE 1.46776943, best enet-1e-3-a25 1.46814867. Promote top two per family as documented. 17/54 measured. Next L05/L06 Huber and median regression, seven predeclared configs. Implement faithful robust loss solvers with finite iteration/convergence gates and synthetic tests, no evaluation-driven changes. Same 2019 cohort. No human action required.

## C004 phase 5
L05/L06 robust ADMM screen implemented, exact Huber and median prox, 1000 iteration cap, 1e-4 primal/stationarity tolerance, unpenalized intercept, four synthetic tests; seven independent CI jobs. Next check CI, archive all successful/blocked outcomes; then latent regressions L07–L09. No scores yet; 17/54 measured. No test data access.
