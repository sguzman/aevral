# C013 — Frozen cross-family ensemble protocol

## Scope and budget

Use existing matched predictions only; no new base-model fits, raw-data changes, domains or 2023–2026 selection. Exactly 24 predeclared combination recipes × two assessment years = 48 combination-fold outcomes maximum. No adaptive extra candidates. Frozen base recipes use seed 42; results compare those exact realized predictions, not C012's seed averages. Eight base columns, ordered as follows:

| Alias | Source |
|---|---|
| I | C003 full-training incumbent mlp64, SGD seed42 |
| L | C012 O04-04 LightGBM |
| R | C012 O05-02 local ridge |
| M | C012 O01-06 optimized MLP80 |
| S | C012 O02-06 suffix24 MLP (matched only) |
| A | C012 O06-06 MARS |
| H | C012 O08-06 Haar ridge |
| X | C012 O07-06 ARX |

All C012 sources are budget 32768, seed42, final rung. N-HiTS excluded before scoring because its seed42 2021 result was runtime-blocked and the cohort is incomplete. No replacement or extra fit. Base shortlist includes full-training incumbent as a distinct comparator, not a budget-equivalent model. C003 daily persistence is a reference only, not a combination column.

## Chronology and artifact gate

C012 recipe selection used 2019; exclude 2019 from C013 calibration and evaluation. For assessment 2021 fit all learned blend parameters exclusively on annual 2020 out-of-fold predictions. For assessment 2022 fit exclusively on 2020–2021 out-of-fold predictions. The underlying base models were trained on earlier observations only, with full target windows within their prediction years. This is chronological development evaluation, not a fresh independent test. All 24 recipes evaluated on the same 2021/2022 matched cohorts (7501 and 7639 origins, 24 leads each). No adaptive selection on either assessment year except that 2021 is allowed calibration for 2022 under the fixed algorithm. Static recipes need no calibration but use identical assessment cohorts. Compare with all eight individual base models and daily persistence on the same exact rows. A-derived S uses matched A cohort; no natural-coverage claim for mixed ensembles.

Download exactly 3 C003 artifacts (run 45892792311, c003-2020/2021/2022) and 21 C012 artifacts (run 46085630766, c012-final-CONFIG-YEAR-42 for seven selected configs and three years). Metadata probes confirmed selected artifacts available; full inventory/download verification pending. Use GitHub CI token with actions:read, no user action, no local download. Freeze inventory by artifact ID, source run and SHA, compare C012 exported prediction hashes against archived log evidence and verify C003 scores against independent export audit. Validate all archives' exact fold, model, origin, lead, finite target/prediction and ordered matched origin SHA256 against C012 cohort-hashes.csv; reject missing, duplicate or misaligned rows and nonidentical target values (numeric tolerance 1e-8 °C for serialization). Recompute base MAE/RMSE against committed scores within 1e-7. Missing or expired artifact is explicit blocker; no substitute forecasts. Cache/retain joined inputs and output artifacts within CI retention; persist manifests, checksums, aggregate evidence in Git. No synthetic data as measured result.

## Fixed recipe matrix

Notation T=[I,L,R], U=all eight columns, Q=[I,L,R,M]. Equal mean and median are computed per origin/lead, even-size median averages middle two. Fit only on calibration rows; all predictions and targets in Celsius. MSE objective for learned recipes fixed before assessment, MAE primary reporting metric. No evaluation labels enter fit.

| ID | Method | Columns |
|---|---|---|
| E01 | Mean | T |
| E02 | Median | T |
| E03 | Mean | U |
| E04 | Median | U |
| E05 | Mean | I,L |
| E06 | Mean | I,R |
| E07 | Mean | L,R |
| E08 | Mean | Q |
| E09 | Mean | A,H,X |
| E10 | Mean | I,S |
| E11 | Global convex | T |
| E12 | Lead-specific convex | T |
| E13 | Global convex | U |
| E14 | Lead-specific convex | U |
| E15 | Global ridge residual stack | T |
| E16 | Lead-specific ridge residual stack | T |
| E17 | Global ridge residual stack | U |
| E18 | Lead-specific ridge residual stack | U |
| E19 | Per-lead residual bias correction | I |
| E20 | Per-lead residual bias correction | Mean(T) |
| E21 | Spread-gated convex | T |
| E22 | Seasonal convex | T |
| E23 | Diversity-aware equal blend | U, select three |
| E24 | Lead-specific calibrated convex | T |

Convex weights: minimize mean squared residual with nonnegative sum-one weights; initialize uniform, projected gradient on simplex using deterministic sorted threshold projection (ties by column order), 2000 iterations or max absolute change <=1e-10. Work in difference coordinates numerically by subtracting I prediction (or first column if I absent) from every column and target; objective unchanged on simplex. Step = 1/max(1e-12, 2*trace of mean feature outer product), a safe Lipschitz bound. Store weights, objective, iterations and convergence status; fixed cap output is valid if finite and feasible, report nonconvergence. Global pools all leads, lead-specific fits each lead separately.

Ridge residual stack: anchor on I; regress target-I on each other prediction-I plus unpenalized intercept, minimize mean squared residual plus 0.1 times sum squared slopes. Center calibration features and target residual, solve positive definite regularized normal equations using stable Cholesky, recover intercept. No unconstrained clipping. Global or lead-specific as specified. Residual bias correction E19/E20 adds calibration mean(target-anchor) per lead. E24 fits E12 then adds half the calibration mean residual per lead (fixed shrinkage).

E21: classify each row by max(T)-min(T) prediction spread above calibration median versus <= median (median over all leads/origins); fit separate global convex weights for each group; if fewer than 1000 calibration rows in either group, use E11 for that group and log fallback. E22: season from first predicted UTC hour (origin+60 minutes), DJF/MAM/JJA/SON; fit global convex weights within each season with same 1000-row fallback. Both gates are inference-time observable, no target input at prediction time. E23: select first column with lowest calibration MSE, then greedily add column minimizing MSE of equal-weight blend of selected plus candidate; stop after three, ties by fixed column order. This is diversity-aware through residual cancellation in joint MSE, not explicit correlation regularization; report calibration pairwise residual correlations and chosen members.

## Verification and report

Rust-first, dependency-free combination/scoring implementation, locked Nix shell for development; GitHub-hosted CI. Unit tests for origin alignment, fold leakage rejection, simplex feasibility, analytic ridge/bias examples, regime boundary/fallback, scoring parity and selection ties. All 48 outcomes or explicit blocks accounted for. Independent recomputation of predictions' overall, lead/month errors and exact keys. Report actual MAE/RMSE, fold differences, all 24 leads and 12 months, base-model costs from recorded runs plus incremental blend fit/predict time and memory. Do not compare cached blend-only runtime to end-to-end base-model time; distinguish shared base reuse and theoretical summed deployment cost. Report complete-cohort rankings, improvement versus best matched single model, weight stability, negative results and limitations. Preserve all source IDs and hashes, coefficient/selection records and implementation SHA. Stop at cap, then C014. No requirement to outperform individual models.
