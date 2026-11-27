# Fixed-data temperature program — C003–C011 complete; C012 active

## Current continuation override (2026-11-27)
C012 R2 run 46083595760 (code 163c5b9f62d593847bc95bbe12c76ffa26278808): prepare and all 24 evaluation jobs succeeded. 14/24 R2 measured results and original filtered logs archived; latest batch O06-06, O05-02, O03-06 registry commit db06882dec9872c9792001d98b58c15161376483. Registry verified 62 unique 24-field entries; 10 R2 remaining, no new fits. Ingest in batches of at most three to avoid tool call cap; check registry job IDs AND existing archive files before creation (interrupted batches can leave logs without registry rows). For each job verify exact request, cohort 7455, 24 leads/12 months, matched rescore and O02 natural rescore, audit pass, 207 tests, resource times and checksums; score has optional extra fields, use indices 3..5 only. Registry schema as existing, R2 budget8192; preserve raw CI provenance. Best archived R2 O05-02 MAE 1.37195009 (partial). After all 24 reconciled freeze R3 one per family and validate before launch. No human action.

Current checkpoint: C012 active; frozen candidate grid and manifest committed before implementation/scoring. Eight representative families, six configs each, 48/24/8 successive-halving rung caps (2019 seed42; 2048/8192/32768 matched fit rows), up to 48 additional finalist annual fits, total <=128 scored outcomes inclusive. Zero C012 outcomes yet. Shared Rust optimization grid/request parser, chronological subset and purged early-stop split implemented with five unit tests; shared infrastructure regression CI 46079779285 passed (196 tests, release build). Implemented native deterministic ARX, Haar, MARS and local ridge parameterized entrypoints (24 frozen configs) and dispatcher; old commands retained, local preflight train-only. Four native integrations passed regression CI 46080030582 (196 tests and build). Added four native targeted tests, strict LightGBM parameter decoding, shared CLI args preserving legacy defaults, exact subset official 24-lead integration, train-only disposable probe, and LightGBM config test; five families/30 configs implemented, regression CI launched. LightGBM combined CLI fit/pred cost explicitly labeled; conservative probe may block expensive configurations. Neural optimizer primitives added: mutable Adam LR, Dense mean batch gradients with optional inverted hidden mask and weight-only L2, N-HiTS mean MAE batch gradients with existing clip convention and weight-only L2; three new primitive tests. LightGBM/native tests CI 46080267584 compiled, 200 passed/1 failed: synthetic ARX fixture rank-deficient (12h timestamps collapse calendar rank); fixed test fixture only to use varied calendar times and deterministic pseudorandom independent predictors. New combined regression triggered; neural primitive CI 46080382076 is superseded for final verification. Combined regression 46080419208 failed compilation on ambiguous Dense gradient types; added explicit f64 and Vec<f64> annotations. Implemented neural config decoder and training orchestration for all 18 neural candidates (SGD/Adam, deterministic shuffles, cosine, minibatches, dropout, decay, purged validation checkpoint and accounting); three targeted tests. Combined regression 46080528875 passed 207 tests and locked release build, including corrected ARX fixture and neural primitives/training. Common C012 CLI now implemented: optimization DATA_DIR FOLD OUTPUT.csv CONFIG_ID RUNG SEED; exact matched/fit origin exports, common cohort/baseline checks, train-only neural preflight and matched/natural suffix24 scoring, actual fit/internal validation origin exports and accounting. Scoring CLI CI 46080661160 failed on missing format placeholder (cfg.stop); corrected. Added C012 independent audit shell (origin hashes from C011, exact subset, neural accounting/purge, matched/natural export rescore, GNU time preservation), frozen R1 matrix of 48, and pinned LightGBM synthetic smoke workflow; regression and smoke launched. Audit shell and scoring workflow not yet end-to-end verified. Prerequisites verified: regression 46080801118 passed 207 tests/build; pinned LightGBM smoke 46080801233 passed. Added train-only ARX/MARS probes and independent month score verification, R1-only six-way CI workflow triggered solely by a separate launch file (not yet created). Final prelaunch CI 46081043967 passed 207 tests, shell syntax and release build. Rung 1 launched by creating launch.txt containing 1 (48 frozen config IDs, 2019 seed42, 2048 selected rows; max six parallel; <=30min each). Rung 1 CI 46081178206 found; prepare passed, six running/42 queued at initial check. Created results/C012 registry schema and standalone Rust promotion validator (incomplete/duplicate/invalid result gates, MAE/RMSE/cost/ID ranking, annual finalist matrix), tests pending. Audit and archive every completed evaluation including explicit blocks before freezing R2; do not relaunch successful audited jobs. Next check R1, reconcile original logs and independently verify promotion tests; update workflow to enforce generated promotions before later launches. 48/48 R1 outcomes archived (48 measured, 0 blocked); provisional best O04-02 MAE 1.44755789. All 48 R1 CI jobs succeeded; remaining 0 awaiting archival. R1 reconciled 48 distinct configs, 48 measured, zero blocked. Frozen rung-2.csv with 24 candidates (top 3 per family by manifest ordering), full R1 ranking persisted. Rust promotion validator CI triggered; later-stage prepare job now independently compares generated matrix to frozen file. Do not launch R2 until promotion CI passes. C011 closed and immutable. No human action.


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

## C004 phase 6
All seven L05/L06 jobs passed CI 45896640395; 45 tests, all converged, 331 records archived and registry updated. Best huber-0.25 MAE 1.46480324, best median-1e-4 1.46546481. Promote top two each as documented. 24/54 measured. Next implement L07–L09 PCR, PLS and reduced-rank regression (12 configs), training-only centered covariance/eigendecomposition, explicit numerical tests and no test data. No human action required.

## C004 phase 7
L07–L09 implemented 12 configs, train-only Gram shared with sparse regression, Jacobi eigensolver residual/orthogonality checks, PCR, covariance PLS2 regression deflation, RRR fitted-response spectral projection; synthetic tests. CLI and CI pending verification. 24/54 measured. Next check latent screen results, archive or repair, then L10–L12 polynomial/GAM/MARS. No new data.

## C004 phase 8
L07–L09 passed CI 45897005852, 48 tests, all 12 measured, 465 result records archived. Best PCR64 1.46788769, PLS32 1.46782162, RRR16 1.46770656; RRR8 1.46814017. Promote top two per family as documented. 36/54 measured. Next L10 polynomial and L11 GAM six configs each, then L12 MARS six configs, preflight workspace/time per manifest. Full polynomial 3320 basis slopes (80 linear + 3240 degree-two monomials) may exceed compute budget; predeclare faithful restricted recent-weather interaction basis and test full basis resource feasibility or explicitly block before fit. No new data. Multi-year finalists pending. No human action required.

## C004 phase 9
L10/L11 implemented 12 planned configs, full/restricted degree-two and additive 3/7 knot splines, train-only expansion, streamed Gram, ridge numerical checks, timed preflight/resource guard, synthetic tests, four CI jobs. 36/54 measured prior to batch. Next verify scores or explicit resource blocks, archive; L12 MARS adaptive hinges/pruning remaining, then multi-year finalists. No new data.

## C004 phase 10
Three basis jobs passed CI 45897341015: poly-recent, gam-3, gam-7; 51 tests; nine scores archived in results/C004, 372 records. Best restricted polynomial 1.33658679, GAM 1.45231366. Full poly job 140174402045 still running; archive remaining three scores or blocks using ORIGINAL code 77848bc72e0a025eb2fd94c4a6ca8fe0d776eb82; don't duplicate the nine already archived. 45/54 measured. Next finalize full polynomial, then L12 MARS. GAM top two actually gam-7-1e-3 and gam-3-1e-3; check exact ranking against gam-7-1e-2 (1.45583124 vs 1.45551824). No new data.

## C004 phase 11
Full polynomial job 140174402045 still running at check; its original code 77848bc72e0a025eb2fd94c4a6ca8fe0d776eb82. L12 bounded-grid MARS implemented six configs, adaptive reflected hinge pairs, train-only sample forward search, full-training refit and backward exact RSS deletion/GCV selection; tests/CI pending. Explicit approximation in L12.md. Five parallel MARS jobs maximum. 45/54 measured, 3 full polynomial + 6 MARS unresolved. Archive both runs without duplicating results; then promote and evaluate finalists. No new data.

## C004 phase 12
All six MARS jobs passed 45897731820, 53 tests, 816 records archived; best mars-grid-2-64 MAE 1.43531178, second mars-grid-1-64 1.47772393. 51/54 measured. Full polynomial job 140174402045 still in progress at last check; archive remaining three results using original 77848bc72e0a025eb2fd94c4a6ca8fe0d776eb82. No duplicate runs. Then freeze top two per method (OLS one) by 2019 MAE, run 2020–22 finalists with <=6 parallel and 30min/job; record all results. No additional data or models.

## C004 phase 13
Full polynomial still active at check. Froze 21 finalists from 11 completely screened methods (L01–L09,L11,L12) based solely on 2019 registry; see campaigns/004-linear-additive/FINALISTS-A.md. Launched grouped 2020–22 evaluation workflow, 18 jobs/63 additional fits, max five concurrent while full poly runs. No L10 promotion until its 2019 results resolve; separate finalists B then. Verify all jobs and archive each completed group without duplication, preserve code SHA and per-config records; report blocked/nonconverged explicitly. Reuse 2019 scores, no fresh holdout claims. Screening remains 51/54. No new data.

## C004 phase 14
Full quadratic CI 45897341015 passed, three results archived with original code SHA, all 54 screens measured. New 2019 leader poly-full-1e-3 MAE 1.32019462, RMSE 1.75294745 vs MLP64 1.32617622/1.76544057; only development. Frozen L10 finalists poly-full-1e-3 and poly-full-1e-2, FINALISTS-B.md. B workflow three sequential annual jobs, two configs sharing Gram per fold; added comma-separated selection to basis CLI only. A CI 45898046310 at check 8/18 completed, 6 running, 4 queued; none archived yet. A logs duplicate score lines via grep: deduplicate exact score/blocked rows per job before registry; preserve lead/month records, fit and RSS per model. Next archive completed A jobs in batches <=6, verify B and archive as completed, then pooled 23-model ranking. Do not duplicate 2019. No test years or new data.

## C004 phase 15
Finalists B CI 45898314791 queued. Finalists A CI 45898046310: archived first 6 completed groups (20 outcomes, 1273 log records) into results/C004/finalists-a-YEAR-GROUP.txt and registry. Deduplicate by job ID registry column 11, exact duplicate score/blocked lines in logs; verify all 24 leads for each measured model. Continue remaining A jobs in batches <=6 and B as complete. No new fits launched this iteration.

## C004 phase 16
Archived next 6 completed A groups (21 outcomes, 1320 records). Total additional outcomes archived 41/69. Continue remaining A and B groups; check registry job IDs for duplicates. No additional fits launched. Part B CI 45898314791.

## C004 phase 17
Archived 5 further A groups (18 outcomes, 1175 records). Additional outcomes archived 59/69. 17 models have complete four-year results. Continue any remaining A and B groups, then pooled final ranking. Part B CI 45898314791. No new fits launched.

## C004 phase 18
Part A all 18 jobs passed; archived final 2022 robust group (4 outcomes, 193 records). 63/69 additional finalist outcomes archived. 21 models complete four-year results, best Part A mars-grid-2-64 pooled MAE 1.48274195. Part B CI 45898314791 still running sequentially; only six polynomial outcomes outstanding. No new fits launched. Final pooled ranking after B completes.

## C004 phase 19
2020 polynomial job 140177628610 passed 53 tests and both frozen fits. Archived 88 log records and 2 outcomes. 65/69 additional outcomes archived. Poly-full-1e-3 2020 MAE 1.31405305 vs incumbent 1.30737647; 2019 poly leads, 2020 neural leads. Part B 2021 job 140177628161 running, 2022 job 140177628573 queued. Archive remaining four outcomes, then calculate pooled ranking and close C004. No new fits launched.

## C004 phase 20
2021 polynomial job 140177628161 passed 53 tests and both frozen fits. Archived 88 records and 2 outcomes. 67/69 additional outcomes archived. Poly-full-1e-3 2021 MAE 1.40918983 vs incumbent 1.39099523. Part B 2022 job 140177628573 running; only two outcomes pending. Archive and calculate final pooled ranking, then close C004. No new fits launched.

## C004 phase 21 — accuracy ranking complete
Part B all jobs completed successfully; 2022 both polynomial fits explicitly blocked by predeclared 1400s conservative preflight (1402.636s). No retry/guard relaxation. Archived final 13 records. Registry 123 outcomes (121 measured, 2 blocked), 54 screens and 69 additional outcomes, all unique, cohort counts checked. REPORT.md and pooled.csv contain 21 complete non-neural models plus incumbent; partial matched 2019–21 comparison separately. MLP64 remains best complete model, best non-neural mars-grid-2-64. C004 accuracy results complete under budget; before declaring final closure verify report resource accounting/manifest and update master campaign index if required. Next C005 statistical/state-space finite manifest. No new data.

## C004 closed; C005 phase 1
Closure audit persisted at results/C004/closure-audit.md; master program and campaign README updated. Registry 123 unique outcomes, 41 log files/job IDs, sample eight logs checked; no independent full prediction recomputation. C005 active, predeclared 36 configs in campaigns/005-statistical-state-space/manifest.md, including 34 point and 2 variance-only. No C005 fits yet. Next implement S01–S04 genuine AR/ARX/VAR(1)/VARX(1), predeclare exact batch details, tests, launch 2019 CI. Retain same fixed cohort; origin weather frozen for ARX, VAR uses joint predicted weather with adjacent training origins. No extra history, no new data.

## C005 phase 2
Implemented S01–S04 ten configurations, recursive AR/ARX/VAR/VARX, streamed QR, causal known calendar and fixed origin weather for ARX, joint VAR state recursion, adjacency guard; four synthetic tests, three CI jobs. S01-S04.md batch specification. Compilation and measured results pending. Next verify/repair CI, archive all ten outcomes with diagnostics, then S05–S07 genuine conditional ARIMA variants. No new data.

## C005 phase 3
S01–S04 CI 45900349024, code efa95b5624c26666fb3e9cbf2dbb4b2e16a0e3ff: archived 2 completed jobs, 6 outcomes, 414 records. 57 tests passed per successful job. 6/36 measured. screen (arx-all) 140184313394 pending Registry initialized; do not duplicate archived jobs. Next collect any remaining autoreg outcomes, then implement S05–S07 conditional ARIMA/ARIMAX/SARIMA; no new data.

## C005 phase 4
All three S01–S04 jobs passed, 57 tests, 10/36 measured; last ARX job 140184313394 archived 346 records. Best ARX72 1.53087031. Frozen provisional finalists AR72/48, ARX72/48, VAR1, VARX1. Next implement S05–S07 nine genuine conditional ARIMA models, no new data.

## C005 phase 5
S05–S07 nine conditional ARIMA/ARIMAX/SARIMA configurations implemented; bounded train-only sample objective, stationarity/invertibility via parameter bounds, seasonal cross terms and integration, causal zero future innovations. S05-S07.md specifies approximate conditional ARMAX/finite optimizer; five tests and nine CI jobs pending. 10/36 previously measured. Next verify/repair, archive nine outcomes and convergence/capped flags, then S08–S11 smoothing and state-space. No new data.

## C005 phase 6
All nine ARIMA CI jobs 45900629055 passed. Archived six outcomes and 282 records, 16/36 measured total. Remaining three jobs (arima-211, arima-201, sarima-101-100) need ingestion, check registry for existing IDs. ARIMAX111 hit optimizer cap; recorded as bounded approximate fit, not converged. Next archive remaining three then S08–S11. No new data.

## C005 phase 7
Remaining three ARIMA outcomes archived from CI 45900629055, 141 records. 19/36 measured. Provisional finalists S05: arima-101, arima-211; S06: arimax-101, arimax-111; S07: sarima-101-010, sarima-101-100. Next implement S08 additive ETS/Holt-Winters and S09 harmonic regression with AR errors, then S10–S11 genuine Kalman state-space, S12 regime switching, S13 GARCH. No new data.

## C005 phase 8
Implemented S08–S09 eight configs, conditional additive ETS incl damped/seasonal and two-stage harmonic regression with recursive AR residuals. Predeclared grid/sample in S08-S09.md; shared QR visibility only. Five tests, eight CI jobs pending. 19/36 measured beforehand. Next verify/repair, archive results; S10–S11 structural/Kalman, S12 switching, S13 GARCH pending. No new data.

## C005 phase 9
All eight S08–S09 jobs succeeded (CI 45901037832); first five archived, 232 records, 24/36 screen outcomes. Next archive remaining three jobs by registry ID, then S10–S11 structural models/Kalman. No new data.

## C005 phase 10
Remaining three S08–S09 outcomes archived from CI 45901037832, 140 records. 27/36 measured. Provisional finalists S08: ets-aaa, ets-ann; S09: harmonic-2-ar1, harmonic-1-ar1. Next implement S10–S11 structural time series and Kalman dynamic linear models (five configurations), then S12 switching and S13 GARCH. No new data.

## C005 phase 11
S10–S11 five Gaussian structural/Kalman configurations implemented, causal fixed-window filtering, train-only NLL grid, PSD check, one-harmonic structural season explicitly documented in S10-S11.md. Five synthetic tests and five CI jobs pending. 27/36 previously measured. Next verify and archive, then S12 Markov switching and S13 GARCH. No new data.

## C005 phase 12
Archived 4 S10–S11 outcomes and 184 log records; 4/5 Kalman outcomes, 31/36 total screen outcomes. CI 45913783874. Next collect unarchived Kalman jobs, then implement S12 Markov switching and S13 GARCH. No new data.

## C005 phase 13
All five Kalman jobs passed; final dlm-trend-harmonic archived (46 records) from CI 45913783874, 32/36 measured. Finalists {"S10":["struct-season","struct-level"],"S11":["dlm-harmonic","dlm-trend-harmonic"]}. Next implement S12 two Markov-switching AR models and S13 two GARCH conditional variance models; separate variance metrics from point ranking. No new data.

## C005 phase 14
Implemented S12/S13 switching and GARCH four configurations with train-only bounded approximate EM, exact regime-weighted first moment, GARCH conditional variance recursion and AR impulse propagation. Batch spec S12-S13.md. Four numerical tests and four CI jobs pending. 32/36 previously measured. Next verify/repair and archive all four outcomes, separate variance metrics; freeze finalist manifest and annual CI. No new data.

## C005 phase 15
All 36 screens measured; final four archived (238 records) from CI 45914013039, 76 tests. Registry audited unique 36 model rows, all 7455 origins. Separate variance.csv. Frozen finalists 22 point + 2 variance-only in FINALISTS.md, 72 additional annual outcomes planned. Next implement grouped annual workflow and launch CI (<=6 parallel); archive all results, pooled ranking and closure audit. No new data.

## C005 phase 16
Annual finalist workflow and grouped runner committed. 24 frozen configs × 3 folds = 72 additional outcomes, 15 jobs, max six parallel, 30-minute timeout. Script cross-checked against FINALISTS.md, exact score/blocked and 24 lead records required, GARCH variance records checked and exported. CI status pending; next verify and ingest completed groups into registry and variance.csv without duplication. Archive resources and optimizer flags. Pool 22 point finalists plus C003 incumbent separately from two uncertainty models, audit closure. No new data.

## C005 phase 17 (annual ingestion)
CI 45914280432: all 15 jobs succeeded. Archived 5 additional jobs, 26 outcomes, 1825 log records. 26/72 additional annual outcomes durable. Registry and variance.csv updated; continue ingestion by unarchived job IDs, then pooled ranking and closure audit. No new data.

## C005 phase 18 (annual ingestion)
CI 45914280432: all 15 jobs succeeded. Archived 5 additional jobs, 24 outcomes, 1133 log records. 50/72 additional annual outcomes durable. Registry and variance.csv updated; continue ingestion by unarchived job IDs, then pooled ranking and closure audit. No new data.

## C005 phase 19 (annual ingestion complete)
CI 45914280432: all 15 jobs succeeded. Archived final 5 jobs, 22 outcomes, 1413 log records. All 72 additional annual outcomes durable, 108 unique measured registry entries with exact cohort counts. Next calculate pooled ranking and separate uncertainty report, audit diagnostics and closure; then C006. No new data.

## C005 closed; next C006
All 108 outcomes audited, 44 job logs present. Final report, pooled.csv and variance-pooled.csv persisted. Best statistical arx-72 MAE 1.59560910 vs MLP64 1.34781461; GARCH evaluated separately. No new data or further C005 fits. Next read C006 scope, freeze finite manifest of faithful tree/ensemble/boosting representatives and then implement first batch. C003–C005 complete; C006–C015 pending. Keep pinned Rust/Nix and hosted CI budget. 

## C006 phase 1
C006 active; predeclared 28 screening configurations across nine tree/boosting families in campaigns/006-trees-boosting/manifest.md. Fixed 24,576 train-origin budget, full eval cohorts, seed policy, official branded backends require version-pinned Nix/CI verification. No C006 fits yet. Next implement T01 exact CART shared-output and per-lead variants with deterministic splits, minimum leaf and resource guard, numerical tests, then launch 2019 CI. No new data.

## C006 phase 2
T01 six exact CART configurations implemented (shared and independent scalar per lead), train sample fixed, numeric SSE splits, 1200s preflight/fit guard, five tests. T01.md batch spec. CI pending; verify compilation and archive six outcomes including any explicit blocks, then T02–T04 ensembles. No new data.

## C006 phase 3
T01 CI 45914941755 succeeded, 81 numerical tests. Six measured screens archived in results/C006 (registry.csv + six logs, 282 records), exact matched cohort counts and lead count verified. Best cart-lead-d8-l32 MAE1.61833904 RMSE2.14853219 fit18.287183s, best shared cart-shared-d8-l32 MAE1.74356087. All within guard. Next implement T02–T04 bagging/RF/Extra Trees using shared CART and deterministic seed, bootstrap and random feature selection per node, independent random thresholds for Extra Trees. 6/28 screens measured. No new data.

## C006 phase 4
T02-T04 ten ensemble configurations implemented, shared 24-output trees, fixed seed42 screening and bounded reproducible SplitMix64, bootstrap/exact split RF/random threshold Extra Trees as manifest. T02-T04.md specifies details. Five new tests. CI pending; verify all jobs, archive outcomes, then T05/T09 histogram boosting. No new data.

## C006 phase 5
T02–T04 CI 45915256233 all ten successful, 86 tests passed. All ten outcomes archived with 470 diagnostic records, exact cohort/lead/seed/sample checks. 16/28 screens measured. Best bag-32 MAE1.54528611 RMSE2.00965579, rf-32-27 1.55554826, extra-32-27 1.65498500. Next T05/T09 histogram squared-loss and median pinball boosting (six configs); train-only quantile bins, per-lead median residual leaf updates for T09. Then official backends T06–T08. No new data.

## C006 phase 6
T05/T09 six configs implemented in boosting.rs, histogram squared-loss and median pinball with median residual leaf updates. T05-T09.md frozen specification. Six new tests and CI workflow submitted; verify compile/test and archive outcomes; then genuine external backends T06-T08. 16/28 measured, 22/28 implemented. No new data.

## C006 phase 7
T05/T09 CI 45916492859 all six successful, 92 tests passed. Archived all six outcomes with 576 diagnostic records, exact cohort/lead/sample/stage checks and objective decrease verified. 22/28 screens measured. Best hist-64-d5 MAE1.40667273 RMSE1.85917069 fit6.765200s; best median-boost-64 MAE1.50829159. Next T06–T08 genuine external XGBoost/LightGBM/CatBoost backends: inspect manifest and pinned Nix availability, no imitations, explicit blocks only if justified. No new data.

## C006 phase 8
Submitted hosted Nix backend availability preflight, per upstream backend (XGBoost, LightGBM, CatBoost), exact flake.lock revision, top-level and python3Packages derivation discovery, bounded realization and CLI/library inspection. No Python runtime introduced, no fitting. Next inspect CI probe logs, select feasible official CLI/library interfaces or separately pinned upstream fallback and freeze versions/configs before fit. 22/28 screens measured. No new data.

## C006 phase 9
All three backend probes succeeded (CI 45916805742). Verified official top-level Nix versions XGBoost3.0.5 shared library, LightGBM4.6.0 CLI/library, CatBoost1.2.10 CLI/library. Archived probe records and added optional boosting devShell using existing lock. Next implement Rust native XGBoost C API and LightGBM/CatBoost CLI adapters, synthetic fit/predict tests, frozen exact parameters and resource guards; then six screens. Shell integration and fit interfaces not yet tested. 22/28 measured. No new data.

## C006 phase 10
Implemented upstream.rs official XGBoost dynamic C API, LightGBM/CatBoost CLI per-lead adapters, label-free eval export, finite/length checks, model counts, timeout guard. T06-T08.md exact recipes/seed policy. Two unit tests and three locked-shell synthetic smoke jobs submitted. Next verify compilation and actual backend fit/predict; repair defects without changing frozen recipes; then six weather screens. No external weather fits yet. 22/28 measured.

## C006 phase 11
First upstream smoke CI 45917145167: XGBoost and LightGBM passed synthetic fit/predict; 94 tests passed on all jobs. CatBoost fit/calc succeeded but strict header parser failed (header contents not captured). Archived outcome in results/C006/upstream-smoke-first.txt. Corrected CatBoost output request to SampleId,RawFormulaVal and parser to validate exact named columns and sequential numeric row IDs, added parser test and descriptive error. No fit recipe changes. Re-run all three smoke jobs automatically; verify before full weather screens. 22/28 measured.

## C006 phase 12
All three official integration smoke jobs passed (CI 45917374767), 95 tests, CatBoost prediction sample identity parser verified. Archived evidence. Submitted six 2019 weather screening jobs through optional locked Nix boosting shell, seed42, exact frozen recipes and cache hashes; upstream-screen.yml. Next verify and archive measured or explicitly resource-blocked outcomes with per-lead tree/leaf counts, then freeze annual finalists. 22/28 measured; six in CI. No new data.

## C006 phase 13
All six official upstream screens passed (CI 45917562327), 95 tests. Archived 426 records, all 28 screening entries unique/measured with matched cohorts. Best xgboost-128 MAE1.36776638, lightgbm-128 1.36797968, MLP64 incumbent1.32617622. Frozen FINALISTS.md: 20 finalists (12 deterministic,8 stochastic), 124 remaining fold/seed evaluations (2019 seed42 reused, other two seeds also required for 2019). Next create bounded grouped annual workflows, run native groups first then Nix upstream groups sequentially to preserve <=6 fitting jobs, archive outcomes and pooled seed statistics. No new data.

## C006 phase 14
Committed native and upstream annual workflows and common runner, exact 20-model frozen list checked, matrices 17 native jobs/90 outcomes and 14 upstream jobs/34 outcomes. Native triggers on push; upstream dispatch-only after native finishes to keep <=6 fitting jobs. Next check native CI, archive by job IDs with fold/model/seed keys; dispatch upstream only after native fitting jobs finish. No new data.

## C006 phase 15 (native ingestion)
Native CI 45917952848: archived 6 completed jobs, 32 additional fold/seed outcomes, 1914 diagnostic records, 32/124 additional outcomes durable. Registry keys fold/model/seed unique; cohorts, sample, seed and lead records verified. Continue archive by unarchived job ID; dispatch upstream workflow after native jobs finish. No new data.

## C006 phase 16 (native ingestion)
Native CI 45917952848: archived 5 further completed jobs, 22 additional fold/seed outcomes, 1245 diagnostic records, 54/124 additional outcomes durable. Registry keys fold/model/seed unique; cohorts, sample, seed and lead records verified. Continue archive by unarchived job ID; dispatch upstream workflow after native jobs finish. No new data.

## C006 phase 17 (native ingestion)
Native CI 45917952848: archived 3 further completed jobs, 18 additional fold/seed outcomes, 855 diagnostic records, 72/124 additional outcomes durable. Registry keys fold/model/seed unique; cohorts, sample, seed and lead records verified. Continue archive by unarchived job ID; dispatch upstream workflow after native jobs finish. No new data.

## C006 phase 18
All 17 native finalist jobs succeeded (CI 45917952848); final 18 outcomes and 855 records archived, 90/124 additional outcomes durable. Registry 118 entries including screening, unique fold/model/seed. No native resource blocks. Upstream workflow launched via path-specific push trigger (dispatch action unavailable), only after all native jobs completed; 14 jobs/34 outcomes, locked Nix boosting shell. Next verify upstream CI run and archive results, compute pooled per-seed rankings and closure. No new data.

## C006 phase 19 (upstream ingestion)
Upstream CI 45918425331: archived 6 completed jobs, 14 additional fold/seed outcomes, 1012 diagnostic records, 104/124 additional outcomes durable. Registry keys fold/model/seed unique; cohorts, sample, seed and per-lead tree records verified. Continue archive by unarchived job ID; compute pooled per-seed rankings and closure when complete. No new data.

## C006 phase 20 (upstream ingestion)
Upstream CI 45918425331: archived 5 further completed jobs, 14 additional fold/seed outcomes, 1009 diagnostic records, 118/124 additional outcomes durable. Registry keys fold/model/seed unique; cohorts, sample, seed and per-lead tree records verified. Continue archive by unarchived job ID; compute pooled per-seed rankings and closure when complete. No new data.

## C006 phase 21 (upstream ingestion)
Upstream CI 45918425331: archived 1 further completed jobs, 2 additional fold/seed outcomes, 145 diagnostic records, 120/124 additional outcomes durable. Registry keys fold/model/seed unique; cohorts, sample, seed and per-lead tree records verified. Continue archive by unarchived job ID; compute pooled per-seed rankings and closure when complete. No new data.

## C006 phase 22
All 14 upstream finalist jobs passed (CI 45918425331); final 4 outcomes and 290 diagnostic records archived. All 124/124 additional evaluations, 152 total registry entries (28 screens+124 additional), zero blocked, unique keys. Next pooled seed-specific metrics, audit complete finalist matrix and close campaign. No new data.
 
## C006 closure
All 152 unique outcomes audited; full 144 finalist matrix plus eight rejected screens. Pooled ranking and per-seed results durable. All annual jobs passed. C007 next: freeze finite manifest. No raw prediction artifact rescore.

## C007 phase 1
Frozen manifest 41 screens (K01–K11), 22 planned finalists, 82 planned additional outcomes; exact subset KNN/kernel/GP/SVR and explicit FITC/RFF/Nystrom approximations, bounded sample budgets, 2019 screen. No code or C007 results yet. Next K01–K04 implementation in bounded batches, using C003 causal harness.

## C007 phase 2
K01–K03 exact 8192-row subset neighbor heap, deterministic ties, uniform/inverse/Gaussian analog weights, seasonal phase filter and explicit nearest fallback implemented; 5 numerical tests, 12 screen configs through neighbors-screen.yml. Timing preflight and runtime guard, cached causal predictions scored on full cohorts. Await CI; no measurements yet. K04 next after verification.

## C007 phase 3
CI 45919448602: 12/12 neighbor screens archived, 12 measured, 564 diagnostic records, 100 tests passed. Initial batch complete; K04 next. Best analog-k16-inverse MAE 1.58604072. No new data.

## C007 phase 4
K04 local linear ridge implemented with shared checked dense Cholesky (reusable for kernel methods), 12 frozen coordinates, exact 64/128 neighbors, mean-loss lambda .01/.1, unpenalized intercept and 24 RHS, residual checks, 5 additional tests, four screens submitted. No K04 measurements yet. Verify CI before proceeding to K05–K06. No new data.

## C007 phase 5
K04 CI 45919760217: 4/4 resolved, 4 measured, 105 tests passed, 188 diagnostic records archived. Best local-k128-l01 MAE 1.37195009. 16/41 screens resolved, best overall local-k128-l01 1.37195009. Next K05–K06. No new data.

## C007 phase 6
K05 exact centered primal linear KRR (8192 rows, mean-loss lambda) and K06 exact dense RBF KRR (2048 rows, diagonal lambda) implemented; shared 24 RHS Cholesky, residual check, cubic preflight scaling, runtime guard in Cholesky, four new parity/analytic tests. Seven frozen screening configs submitted. No K05/K06 measurements yet. Verify CI before K07 official LIBSVM probe. No new data.

## C007 phase 7
K05–K06 CI 45920051360: 7/7 resolved, 7 measured, 109 tests passed, 329 diagnostic records archived. Best rbf-krr-s1-l1 MAE 1.39084668. 23/41 screens resolved, best overall local-k128-l01 1.37195009. Next K07 pinned official LIBSVM probe. No new data.

## C007 phase 8
Added optional svr devShell using existing flake.lock pin and official pkgs.libsvm, package version/root exposed, CI probe inventories CLI/library paths and tests epsilon-regression CLI if available. No lock update, no changes to default shell. Await actual probe results before implementing adapter. 23/41 measured; no new data.

## C007 phase 9
LIBSVM 333 pinned package probe passed (CI 45920309889, CLI support vector count4 and expected predictions); results/C007/libsvm-probe.txt durable. Implemented 24 separate scalar epsilon-SVR fits, official CLI absolute paths, dummy zero eval labels, model/header/count validation, finite predictions, first full lead preflight (reused), timeout guard, three parser/export tests. Smoke CI before four frozen weather configs svr-s05-c1/c10 and svr-s1-c1/c10. 23/41 measured; no new data.

## C007 phase 10
Official LIBSVM adapter smoke passed CI 45920563721, 112 tests, synthetic predictions -1.04999996/1.04999996, four support vectors. Evidence results/C007/svr-smoke.txt. Four weather screens launched via svr-screen.yml with locked Nix shell, 2048 train rows and full 2019 cohort, fixed C/length pairs. 23/41 measured; await results before K08–K11. No new data.

## C007 phase 11
K07 CI 45920736527: 4/4 resolved, 4 measured, 112 tests passed, 284 diagnostic records archived. Best svr-s1-c1 MAE 1.40649144. 27/41 screens resolved, best overall local-k128-l01 1.37195009. Next K08–K09. No new data.

## C007 phase 12
K08 exact 1024-row GP and K09 8192-row FITC GP with 128/256 inducing points, shared covariance, mean-centered 24 outputs, fixed noise .05/jitter 1e-8. Streaming sufficient statistics, bounded diagonals and latent variance diagnostics first32, residual checks, four analytic tests including exact-inducing limit; six frozen screens launched. No GP measurements yet. 27/41 measured; verify CI before K10–K11.

## C007 phase 13
K08–K09 CI 45921039141: 6/6 resolved, 6 measured, 116 tests passed, 292 diagnostic records archived. Best fitc-s1-m256 MAE 1.37723166. 33/41 screens resolved, best overall local-k128-l01 1.37195009. Next K10–K11. No new data.

## C007 phase 14
RFF SplitMix64/Box–Muller independent feature-major draws and Nystrom deterministic landmarks/triangular map implemented, centered feature ridge .01 and 8192 reference rows, train-only 64-pair kernel RMSE diagnostic, four numerical tests. Eight seed42 screens submitted; deterministic Nystrom uses seed42 CLI placeholder. 33/41 measured; verify CI before freeze finalists. No new data.
