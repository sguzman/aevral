# C013 — Cross-family ensembles

**Complete assessment:** 24 recipes × two folds = 48 measured outcomes, no blocks. The four-model equal mean E08 (I,L,R,M) leads matched pooled MAE at 1.35048272°C, 2.04% below the best individual source, the full-training incumbent I (1.37861640°C). E13 global convex has lowest pooled RMSE (1.82116213°C). Differences between leading blends are small and descriptive; this is development-period evidence, not independent confirmation.

## Frozen design and provenance

[Manifest](manifest.md), [source inventory](source-inventory.csv), [source hashes](source-hashes.csv). Reuse eight seed42 base predictions and daily persistence, no new base fits. Calibrate 2021 exclusively on 2020 out-of-fold rows, 2022 exclusively on 2020–2021 out-of-fold rows; exclude 2019 (C012 recipe selection). All base predictions verified from historical runs, with strictly past-only training. Incumbent uses full expanding training; seven C012 columns use 32,768-row fit budgets. N-HiTS excluded in advance due to missing seed42 2021 predictions. Assess 7,501 and 7,639 matched origins, 15,140 total, 363,360 targets. The A-derived suffix model contributes only matched predictions. All fits complete before current assessment file loads. No 2023–2026 data or fresh holdout claim.

## Results

MAE rank over all 33 scored columns (24 blends, eight individual models and daily). IDs defined in manifest; tied rounded scores retain source order. Pool MAE by origin count and RMSE by square root of origin-weighted squared annual RMSE.

| Rank | ID | MAE °C | RMSE °C | 2021 MAE | 2022 MAE |
|---:|---|---:|---:|---:|---:|
| 1 | E08 | 1.350483 | 1.821736 | 1.362852 | 1.338336 |
| 2 | E14 | 1.350681 | 1.821597 | 1.363371 | 1.338220 |
| 3 | E13 | 1.350959 | 1.821162 | 1.364179 | 1.337978 |
| 4 | E12 | 1.351369 | 1.821989 | 1.364702 | 1.338277 |
| 5 | E17 | 1.351560 | 1.821402 | 1.364121 | 1.339226 |
| 6 | E24 | 1.351929 | 1.822145 | 1.364454 | 1.339631 |
| 7 | E18 | 1.352101 | 1.822501 | 1.365573 | 1.338872 |
| 8 | E11 | 1.352113 | 1.822044 | 1.365618 | 1.338852 |
| 9 | E21 | 1.352205 | 1.822100 | 1.365769 | 1.338887 |
| 10 | E01 | 1.352905 | 1.824431 | 1.365624 | 1.340416 |
| 11 | E23 | 1.352905 | 1.824431 | 1.365624 | 1.340416 |
| 12 | E16 | 1.354175 | 1.823738 | 1.366071 | 1.342494 |
| 13 | E22 | 1.354399 | 1.823373 | 1.367114 | 1.341913 |
| 14 | E15 | 1.354410 | 1.823349 | 1.366169 | 1.342863 |
| 15 | E20 | 1.354744 | 1.825525 | 1.366430 | 1.343268 |
| 16 | E05 | 1.358006 | 1.833573 | 1.368961 | 1.347250 |
| 17 | E02 | 1.358570 | 1.831955 | 1.370543 | 1.346813 |
| 18 | E06 | 1.361511 | 1.831878 | 1.376868 | 1.346431 |
| 19 | E07 | 1.370975 | 1.848018 | 1.382842 | 1.359322 |
| 20 | I | 1.378616 | 1.851400 | 1.390995 | 1.366461 |
| 21 | E19 | 1.381489 | 1.853988 | 1.391096 | 1.372057 |
| 22 | E04 | 1.385399 | 1.862473 | 1.390474 | 1.380416 |
| 23 | E10 | 1.386970 | 1.864499 | 1.395781 | 1.378318 |
| 24 | M | 1.393320 | 1.873821 | 1.402856 | 1.383956 |
| 25 | E03 | 1.394649 | 1.872971 | 1.399338 | 1.390044 |
| 26 | L | 1.396612 | 1.887571 | 1.404966 | 1.388408 |
| 27 | R | 1.408653 | 1.893196 | 1.425714 | 1.391900 |
| 28 | S | 1.465893 | 1.959517 | 1.481264 | 1.450799 |
| 29 | A | 1.519007 | 2.040220 | 1.545768 | 1.492729 |
| 30 | E09 | 1.544148 | 2.054551 | 1.549632 | 1.538763 |
| 31 | H | 1.609993 | 2.129849 | 1.604162 | 1.615719 |
| 32 | X | 1.639844 | 2.165455 | 1.631896 | 1.647648 |
| 33 | daily | 2.127019 | 2.817551 | 2.148988 | 2.105447 |

E08 improves incumbent MAE in both years (1.39099523 → 1.36285245 and 1.36646120 → 1.33833646). E13 has marginally better 2022 MAE, E08 better pooled MAE. E01 and E23 select the same three members in both folds, so their scores coincide to reported precision. Nineteen ensemble recipes outperform I in pooled MAE; five do not. Averaging all eight (E03) is worse than incumbent, as is standalone residual correction E19. Diversity and calibration do not guarantee gains.

## Diagnostics

| Horizon | E08 MAE | I MAE |
|---:|---:|---:|
| 1h | 0.54168234 | 0.59973002 |
| 6h | 1.12873121 | 1.16332771 |
| 12h | 1.42266883 | 1.44767675 |
| 18h | 1.60370964 | 1.62250026 |
| 24h | 1.73442835 | 1.75920609 |

June remains challenging (E08 1.67522308, I 1.74216431). E08 improves monthly MAE in 11/12 months; October worsens (1.39530759 vs 1.38743703). All 24 leads and 12 months are archived for every column, including negative outcomes. No significance inference from overlapping targets.

Global convex E13 weights (I,L,R,M,S) change from approximately (0.402,0.174,0.273,0.098,0.053) to (0.387,0.195,0.279,0.123,0.015); A,H,X receive zero in both years. All reported optimizer convergence flags true, no regime fallback. Ridge can assign negative slopes; these are not convex weights. Spread thresholds 0.700449 and 0.715770°C. E23 selected I,R,L both times. Full coefficients, correlations and biases retained.

## Cost and reproducibility

E08 uses four independently fitted models; summed source fit+predict costs are 2021: 190.821s; 2022: 196.988s (not a measured joint deployment runtime). Incremental cached prediction compute 0.006346s and 0.006503s, including per-call clock overhead, not end-to-end latency. Base scores in fold-scores.csv use zero timing placeholders; actual source costs are in source-inventory.csv. Whole assessment scorer measured 2.67s elapsed and 59,032 KiB peak RSS, excluding base-model training, download and separate auditor. Cheap blending is not free base inference.

[Source audit](../../results/C013/source-audit.txt) · [Assessment audit](../../results/C013/assessment-audit.txt) · [Fold scores](../../results/C013/fold-scores.csv) · [All metrics](../../results/C013/metrics.csv) · [Pooled diagnostics](../../results/C013/pooled-diagnostics.csv) · [Coefficients](../../results/C013/coefficients.csv) · [Parameters](../../results/C013/parameters.csv).

[Assessment CI](https://github.com/sguzman/aevral/actions/runs/46090210039) independently verified all 363,360 exported rows, 2,442 metrics, 262 coefficient groups and 228 parameter records. [Archival CI](https://github.com/sguzman/aevral/actions/runs/46090389312) verified all seven output hashes. Prediction artifacts are finite-retention; aggregate evidence and hashes durable in Git. Reproduce pooled results in locked development shell:

```sh
nix develop --command rustc --edition=2021 -O campaigns/013-ensembles/pool-diagnostics.rs -o /tmp/c013-pool
/tmp/c013-pool results/C013/metrics.csv /tmp/c013-pooled.csv
cmp /tmp/c013-pooled.csv results/C013/pooled-diagnostics.csv
```

## Limitations and next steps

Only two assessment years and one fixed seed per source; model selection history is exposed development data. No independent generalization or statistical significance claim, no extrapolation to unseen stations. C015 will freeze finalists before retrospective assessment. C014 addresses distributional forecasts under its own finite protocol.
