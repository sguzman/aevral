# C015 — Shortlist selection (frozen identities)

Selected 2026-11-27 from durable C003–C014 development records, before new 2023–2026 retrospective evaluation. This fixes model identities and source recipes, not yet the complete executable manifest. No assessment outcomes were consulted for this selection. Historical C001/C002 exposure remains.

## Single-model candidates (12/12)

Preserve original fit budgets, objectives, preprocessing and implementations. I is full expanding training; budgeted models sample only eligible prior rows according to their original source recipe.

| ID | Source recipe | Training origins cap | Development pooled MAE °C | Rationale |
|---|---|---:|---:|---|
| I | C003 mlp64 SGD seed42 | Full | 1.34781461 | Mandatory incumbent |
| L | C012 O04-04 LightGBM | 32768 | 1.36663960 | Best complete C012 model |
| R | C012 O05-02 local ridge | 32768 | 1.37500383 | Strong local model |
| M | C012 O01-06 MLP80 | 32768 | 1.37886325 | Optimized dense model |
| S | C012 O02-06 suffix24 MLP | 32768 | 1.42896796 | Short history and coverage |
| A | C012 O06-06 MARS | 32768 | 1.483253 | Fast additive comparator |
| H | C012 O08-06 Haar ridge | 32768 | 1.557362 | Low-cost decomposition |
| X | C012 O07-06 ARX | 32768 | 1.590479 | Classical statistical comparator |
| G | C006 xgboost-128 | 24576 | 1.41090543 | Strong alternative official booster |
| K | C007 fitc-s1-m256 | 8192 | 1.43321565 | Distinct sparse GP family |
| N | C010 mae-nhits64 | 8192 | 1.45243011 | Best complete objective-study recipe |
| W | C011 pressure-tendency-short | 8192 | 1.56114726 | Best B-expanded representation |

Sources: [C003](../../results/C003/README.md), [C006](../../results/C006/REPORT.md), [C007](../../results/C007/REPORT.md), [C010](../../results/C010/pooled.csv), [C011](../../results/C011/annual-rankings.csv), [C012](../012-optimization/REPORT.md). Development values for stochastic recipes are seed means where applicable; I is seed42. Training budgets differ; not a controlled equal-training-data comparison. W is Track B-expanded; S is A-derived. Other candidates use Track A. The eight C013 bases are retained to reproduce learned blend recipes without changing member sets.

N uses the fully measured C010 recipe instead of C012 O03-04: its lower available-case MAE is not a complete-cohort result (three runtime blocks, including seed42 in 2021). Other near-duplicate, lower-ranked or less informative recipes are excluded from the finite final shortlist, not erased from historical ranking. XGBoost and FITC represent alternatives to L and R; N represents structured neural models and W tests the expanded-information track. Daily persistence remains a non-fitted benchmark, not a thirteenth candidate.

## Ensemble and postprocessor candidates (4/4)

| ID | Original recipe | Development rationale |
|---|---|---|
| E08 | C013 equal mean I,L,R,M | Best C013 pooled MAE 1.35048272 |
| E13 | C013 global convex over I,L,R,M,S,A,H,X | Best C013 pooled RMSE 1.82116213 |
| E14 | C013 lead-specific convex over same eight | Second C013 pooled MAE 1.350681 |
| P13 | C014 depth-2 per-lead residual distribution tree over E08 center and I,L,R,M spread | Best C014 CRPS 0.97156891 and median MAE 1.34945390 |

Sources: [C013 report](../013-ensembles/REPORT.md), [C013 manifest](../013-ensembles/manifest.md), [C014 report](../014-probabilistic/REPORT.md), [C014 manifest](../014-probabilistic/manifest.md). P13 counts against the four-combination cap; rank its issued median in the point comparison and separately report proper distributional scores and coverage. No new combinations, probabilistic recipes or hyperparameter trials.

## Chronology and seeds

Four retrospective assessment folds: calendar 2023, 2024, 2025 and Jan–Oct 2026, full 24-hour target windows inside each fold. Full expanding eligible training or original fixed subset budget; fit transforms on allowed training data with exact original recipe semantics. All scores on common matched A cohort; natural coverage separately for S and optional observed history for W. W never silently enters the primary A-only ranking. Primary comparison uses seed42 for all models, including ensemble members; also retain seeds 123 and 2026 for stochastic M,S,N,W as sensitivity, not extra candidate recipes or independent weather observations. I stays the original seed42 incumbent. Deterministic candidates run once per fold. Cap 48 primary single-fold outcomes plus 32 seed-sensitivity fits and 16 combination-fold outcomes, excluding non-fitted baseline metrics. Failures explicitly recorded rather than substituted.

For learned E13/E14/P13, reuse checksum-verified C013 2020–2022 joined out-of-fold forecasts (artifact 15239944097 and six committed hashes). Exclude 2019 selection year. For each new assessment year, fit on 2020 through preceding year only, appending previous retrospective folds' verified seed42 base forecasts under the same fixed algorithm; never fit on current assessment targets or tune after exposure. E08 needs no calibration. No use of observed future weather at inference. Artifact/cache loss is a blocker, not permission to silently alter inputs.

## Next manifest gate

Before launching any assessment, inspect source code/runner identities, fold support and recipe fidelity; freeze exact execution commands, seed matrix, input hashes, origin/cohort rules and independent scoring/audit, block-bootstrap design and resource guards. Preserve previous results. Optional fixed-original-training 2026 comparison only if affordable and explicitly accounted for within approved limits; otherwise document omitted. No assessment executed yet. No prospective confirmation or new data authorized.
