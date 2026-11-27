# C011 — Final same-archive representation study

Complete: 29 screens plus 319 annual evaluations, 348/348 measured, zero blocked. Each recipe has four annual development folds and three seeds, 30,138 matched origins per seed or 90,414 origin-seed pairs (not independent observations). All outcomes reconciled to the frozen manifest, promotion registry and unique CI job IDs. All additional annual jobs passed 191 tests; staged screening test counts are in the screening README. Same 8,192 training origins, 12 epochs, Adam .001, 98,304 updates per fit, original target normalization; width changes parameter count and compute. No new data. Raw prediction payloads were independently rescored inside CI, not downloaded and rescored separately here. CI artifact retention is 30 days.

## Matched development rankings

Weighted MAE and pooled RMSE computed from rounded registry metrics by origin count; tie-break RMSE then total fit+predict time (no ties required). Paired wins/losses compare the same fold and seed to base80; seed SD is descriptive population SD of three pooled seed MAEs, not an iid uncertainty estimate. Compute totals sum 12 evaluations, with variable hosted runner performance; prediction timing includes natural predictions for suffix recipes. RSS is peak of recorded process maxima, not model-only workspace. See [annual rankings](annual-rankings.csv), [paired control](paired-control.csv), [compute costs](compute-costs.csv).

### A-control and A-derived (same origin information)

| Rank | Recipe | MAE °C | RMSE °C | Wins/losses vs base80 | Seed SD °C | Parameters | Fit+predict s | Peak RSS MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | suffix24 | 1.50364851 | 2.00916872 | 12/0 | 0.01555111 | 3672 | 41.45 | 320.9 |
| 2 | suffix48 | 1.55456904 | 2.07248137 | 11/1 | 0.02709973 | 5208 | 59.89 | 320.7 |
| 3 | wind-speed | 1.58987838 | 2.10841675 | 6/6 | 0.01819756 | 6808 | 75.40 | 320.8 |
| 4 | rolling-6-24-72 | 1.59036499 | 2.10179121 | 6/6 | 0.01514185 | 7320 | 78.76 | 320.9 |
| 5 | gradient-long | 1.59100124 | 2.10923125 | 7/5 | 0.01172300 | 7128 | 81.03 | 320.9 |
| 6 | base80 | 1.59212193 | 2.11688675 | 0/0 | 0.02636705 | 6744 | 74.96 | 320.9 |
| 7 | rolling-3-12-48 | 1.59331506 | 2.11042800 | 6/6 | 0.02084899 | 7320 | 86.97 | 320.8 |
| 8 | gradient-short | 1.59482879 | 2.11518936 | 6/6 | 0.02398920 | 7128 | 73.10 | 321.0 |
| 9 | dew-depression-add | 1.59521167 | 2.12131844 | 7/5 | 0.03282222 | 6808 | 74.87 | 320.9 |
| 10 | dew-depression-replace | 1.59904903 | 2.11976646 | 5/7 | 0.01240473 | 6744 | 76.48 | 320.8 |
| 11 | interactions-calendar | 1.60416271 | 2.13137686 | 5/7 | 0.02763154 | 7512 | 83.81 | 320.7 |
| 12 | wind-unit | 1.60611917 | 2.12661647 | 4/8 | 0.03018588 | 7000 | 74.22 | 321.0 |
| 13 | interactions-local | 1.60669801 | 2.12866396 | 7/5 | 0.01933561 | 7128 | 76.24 | 320.8 |
| 14 | spectral-3 | 1.60879069 | 2.13490063 | 3/9 | 0.02049189 | 7128 | 76.74 | 320.9 |
| 15 | anomaly-month | 1.61608729 | 2.14339259 | 3/9 | 0.02408770 | 8280 | 84.68 | 320.8 |
| 16 | harmonics-2-3 | 1.61952493 | 2.13817097 | 2/10 | 0.01640022 | 7256 | 75.41 | 320.8 |
| 17 | spectral-5 | 1.62049929 | 2.14463250 | 4/8 | 0.01256708 | 7384 | 73.62 | 320.9 |
| 18 | harmonics-2-4 | 1.62075260 | 2.14010813 | 2/10 | 0.01447237 | 7512 | 81.61 | 320.9 |
| 19 | anomaly-month-hour | 1.63708230 | 2.17853527 | 1/11 | 0.03103870 | 8280 | 90.84 | 321.1 |
| 20 | select48-leads | 1.75482195 | 2.32266134 | 0/12 | 0.04005802 | 4696 | 50.59 | 321.0 |
| 21 | select32-mean | 1.79335826 | 2.37298727 | 0/12 | 0.04805148 | 3672 | 43.18 | 320.9 |

### B-expanded (additional same-archive history; separate ranking)

| Rank | Recipe | MAE °C | RMSE °C | Wins/losses vs base80 | Seed SD °C | Parameters | Fit+predict s | Peak RSS MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | pressure-tendency-short | 1.56114726 | 2.06614529 | 10/2 | 0.03233047 | 7128 | 80.21 | 321.0 |
| 2 | weather-history-short | 1.56409912 | 2.06262638 | 8/4 | 0.01890035 | 8280 | 75.80 | 320.9 |
| 3 | weather-history-long | 1.56453606 | 2.06130886 | 9/3 | 0.02787039 | 8280 | 95.58 | 321.0 |
| 4 | sparse-lags-14d | 1.56845895 | 2.07751116 | 8/4 | 0.01810925 | 7512 | 80.56 | 321.0 |
| 5 | pressure-tendency-long | 1.57182423 | 2.07734004 | 9/3 | 0.01053885 | 7128 | 69.17 | 321.0 |
| 6 | sparse-lags-28d | 1.59501616 | 2.10308993 | 6/6 | 0.01956205 | 7768 | 96.45 | 320.9 |
| 7 | history96 | 1.61181347 | 2.12379500 | 4/8 | 0.02078371 | 9816 | 104.74 | 320.9 |
| 8 | history168 | 1.76387654 | 2.28880675 | 0/12 | 0.00572961 | 19032 | 203.22 | 321.0 |

## Coverage and variability

| Recipe | Matched origins per seed | Natural origins per seed | Natural MAE °C | Natural RMSE °C | All optional observed per seed |
|---|---:|---:|---:|---:|---:|
| suffix24 | 30138 | 31956 | 1.49962859 | 2.00248310 | n/a |
| suffix48 | 30138 | 31032 | 1.55283842 | 2.06924178 | n/a |
| wind-speed | 30138 | 30138 | 1.58987838 | 2.10841675 | n/a |
| rolling-6-24-72 | 30138 | 30138 | 1.59036499 | 2.10179121 | n/a |
| gradient-long | 30138 | 30138 | 1.59100124 | 2.10923125 | n/a |
| base80 | 30138 | 30138 | 1.59212193 | 2.11688675 | n/a |
| rolling-3-12-48 | 30138 | 30138 | 1.59331506 | 2.11042800 | n/a |
| gradient-short | 30138 | 30138 | 1.59482879 | 2.11518936 | n/a |
| dew-depression-add | 30138 | 30138 | 1.59521167 | 2.12131844 | n/a |
| dew-depression-replace | 30138 | 30138 | 1.59904903 | 2.11976646 | n/a |
| interactions-calendar | 30138 | 30138 | 1.60416271 | 2.13137686 | n/a |
| wind-unit | 30138 | 30138 | 1.60611917 | 2.12661647 | n/a |
| interactions-local | 30138 | 30138 | 1.60669801 | 2.12866396 | n/a |
| spectral-3 | 30138 | 30138 | 1.60879069 | 2.13490063 | n/a |
| anomaly-month | 30138 | 30138 | 1.61608729 | 2.14339259 | n/a |
| harmonics-2-3 | 30138 | 30138 | 1.61952493 | 2.13817097 | n/a |
| spectral-5 | 30138 | 30138 | 1.62049929 | 2.14463250 | n/a |
| harmonics-2-4 | 30138 | 30138 | 1.62075260 | 2.14010813 | n/a |
| anomaly-month-hour | 30138 | 30138 | 1.63708230 | 2.17853527 | n/a |
| select48-leads | 30138 | 30138 | 1.75482195 | 2.32266134 | n/a |
| select32-mean | 30138 | 30138 | 1.79335826 | 2.37298727 | n/a |
| pressure-tendency-short | 30138 | 30138 | 1.56114726 | 2.06614529 | 30138 |
| weather-history-short | 30138 | 30138 | 1.56409912 | 2.06262638 | 27970 |
| weather-history-long | 30138 | 30138 | 1.56453606 | 2.06130886 | 27778 |
| sparse-lags-14d | 30138 | 30138 | 1.56845895 | 2.07751116 | 29846 |
| pressure-tendency-long | 30138 | 30138 | 1.57182423 | 2.07734004 | 30138 |
| sparse-lags-28d | 30138 | 30138 | 1.59501616 | 2.10308993 | 29711 |
| history96 | 30138 | 30138 | 1.61181347 | 2.12379500 | 29278 |
| history168 | 30138 | 30138 | 1.76387654 | 2.28880675 | 26853 |

All natural cohorts except suffix24/48 equal matched cohorts. Optional completeness is not a restriction on matched evaluation: missing optional values are masked. Natural scores with differing cohorts are not paired comparisons. Fold and seed means for every recipe are in annual-rankings.csv; paired delta ranges and descriptive fold/seed SDs in paired-control.csv.

## Decision

- A-derived winner suffix24: 1.50364851°C vs base80 1.59212193°C, 5.56% relative MAE reduction, all 12 paired comparisons improve (delta range -0.15699417 to -0.03442671°C). 32 vs 80 inputs, 3,672 vs 6,744 parameters, 44.71% lower recorded total fit+predict time (runner-dependent). Natural coverage increases from 30,138 to 31,956 origins per seed. Suffix48 is second and wins 11/12 comparisons.
- B-expanded winner pressure-tendency-short: 1.56114726°C, wins 10/12 against matched-budget base80. All optional features observed on matched evaluation origins in this archive. Weather-history-short and -long are close alternatives; their optional coverage is lower. Do not pool B and A rankings.
- Longer history is not reliably useful under this recipe/budget: history96 worsens pooled MAE, history168 worsens all 12 pairs. Correlation-only feature selection at 32 or 48 inputs is substantially worse; the 32-feature temporal suffix is not equivalent to selecting 32 correlation-ranked features. Additional harmonics, anomalies, spectral and interactions do not outperform the suffix recipe. Small differences around base80 are descriptive, not significance claims.
- C003 full-training incumbent development MAE 1.34781461°C / RMSE 1.80956416°C; C010 matched-budget MAE N-HiTS64 1.45243011°C / 1.94294115°C. C011 does not beat either reference. The C003 budget differs and C010 changes model/objective, so these are references, not controlled representation-only comparisons. Historical 2023–26 periods remain exposed and unused for selection in C011.
- Close C011 without additional recipes; retain suffix24 as a representation candidate for C012's separately frozen finite optimization shortlist, not as an adopted independent-test winner.

## Audit and provenance

[Screen README](README.md), [registry](registry.csv), [frozen manifest](../../campaigns/011-feature-engineering/manifest.md). Annual CI [1](https://github.com/sguzman/aevral/actions/runs/46073083514), [2](https://github.com/sguzman/aevral/actions/runs/46074614113), [3](https://github.com/sguzman/aevral/actions/runs/46076864280), [4](https://github.com/sguzman/aevral/actions/runs/46078309340). Archive ingestion omitted wall-clock/RSS labels in 204 records; first report audit and first recovery failed; corrected [audit CI](https://github.com/sguzman/aevral/actions/runs/46079338918) recovered all 204 from original CI logs and checked parameters, fit/predict, wall and peak RSS across all 348 archived jobs. [Recovered timings](timing-recovered.csv) retain original job IDs; no retraining or imputation. Source audit scripts retained in scripts/c011-report.rs and scripts/c011-recover-timing.sh.
