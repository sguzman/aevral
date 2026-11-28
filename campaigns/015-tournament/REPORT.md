# C015 — Final fixed-data temperature tournament

**Complete, 2026-11-27 (local time).** All 96 predeclared evaluations measured and independently audited; zero C015 failures or resource blocks. This closes the C003–C015 fixed-data research program. No prospective confirmation is claimed.

## Research question and design

Which frozen forecasting recipe best predicts the next 24 hourly air temperatures at KSEA, using the existing archive and no new observations? The [manifest](manifest.md), [shortlist](SHORTLIST.md), [fit matrix](fit-matrix.csv) and [combination matrix](combination-matrix.csv) were committed before assessment. The 27 raw-file hashes were verified against the [snapshot](../../results/AEV-015/snapshot.sha256). Expanding chronological training precedes each annual assessment (2023, 2024, 2025 and January–October 2026); no training label crosses the assessment boundary. Scaling is training-only. The historical periods had been exposed by C001/C002, so this is a retrospective comparison, not a new untouched test. The optional fixed-original-training comparison was omitted under the frozen finite budget.

The 80 single-model fits comprise 48 primary and 32 additional seed evaluations. Sixteen combinations use fixed recipes, with calibration restricted to prior-year development predictions. Track A matches the original information set, including the suffix24 derived representation on identical cases; Track B's pressure-tendency representation is separate. All 30,275 matched origins (726,600 hourly targets) are common across ranked candidates. Counts and coverage:

| Fold | Matched origins | Targets | Natural suffix24 origins |
|---|---:|---:|---:|
| 2023 | 8,026 | 192,624 | 8,203 |
| 2024 | 7,730 | 185,520 | 8,099 |
| 2025 | 7,998 | 191,952 | 8,235 |
| Jan–Oct 2026 | 6,521 | 156,504 | 6,755 |
| Pooled | 30,275 | 726,600 | 31,292 |

Matched coverage is 90.35% of 33,508 possible origins; natural suffix24 coverage is 93.39%. Quarantines and missing observations were not filled using future data. Natural suffix24 seed42 MAE is 1.40173°C (different cohort, not ranked against matched results). November and December 2026 are unavailable.

## Primary results

Pooled target-weighted MAE in °C, then RMSE and compute tie-breaks. Seed42 fixed in advance; no seed shopping. All four combinations beat the incumbent; no other single-model candidate does.

| Rank | ID | Recipe | MAE | RMSE | Median fit + prediction (s) |
|---:|---|---|---:|---:|---:|
| 1 | P13 | Depth-two residual distribution tree, issued median | 1.295590 | 1.747140 | 3.677538* |
| 2 | E14 | Lead-specific eight-member convex ensemble | 1.300316 | 1.749265 | 0.123430* |
| 3 | E13 | Global eight-member convex ensemble | 1.301486 | 1.750037 | 0.063721* |
| 4 | E08 | Equal four-member mean | 1.304066 | 1.752657 | 0.049500* |
| 5 | I | Incumbent MLP64 | 1.328789 | 1.781044 | 88.675316 |
| 6 | L | LightGBM | 1.345043 | 1.810273 | 45.451581 |
| 7 | R | Local ridge | 1.354904 | 1.816111 | 35.974404 |
| 8 | M | Adam MLP80 | 1.374273 | 1.827121 | 25.610156 |
| 9 | G | XGBoost | 1.390135 | 1.860593 | 4.906561 |
| 10 | K | Sparse FITC Gaussian process | 1.396043 | 1.860332 | 0.818950 |
| 11 | S | Suffix24 MLP | 1.404680 | 1.870119 | 13.941975 |
| 12 | N | MAE N-HiTS | 1.415439 | 1.899010 | 63.009086 |
| 13 | A | MARS | 1.448800 | 1.922559 | 0.539865 |
| 14 | H | Haar ridge | 1.547792 | 2.040498 | 0.180245 |
| 15 | X | ARX | 1.581405 | 2.085687 | 0.619861 |

*Combination incremental cost only. Daily persistence reference: MAE 2.072207, RMSE 2.707889. Separate Track B: W pressure-tendency short-history MLP MAE 1.493635, RMSE 1.981256, 7.241570 s. P13 reduces incumbent MAE by 0.033199°C (2.50%) and daily persistence MAE by approximately 37.48%. It is the retrospective winner, not a guarantee of future skill.

### Annual robustness

| Fold | I MAE | E08 | E13 | E14 | P13 |
|---|---:|---:|---:|---:|---:|
| 2023 | 1.390495 | 1.352247 | 1.352653 | 1.352233 | 1.340592 |
| 2024 | 1.288818 | 1.258328 | 1.256289 | 1.254871 | 1.251837 |
| 2025 | 1.305505 | 1.285817 | 1.281341 | 1.279603 | 1.274609 |
| Jan–Oct 2026 | 1.328783 | 1.321363 | 1.316795 | 1.315689 | 1.317802 |

P13 beats I on all four folds, but E14 beats P13 on 2026 MAE. The performance gap between combinations is small; no paired P13-versus-E14 significance claim is made.

### Paired uncertainty

Within-fold calendar-block bootstrap, 10,000 common resamples per block length, weighted by sampled origin counts; percentile intervals. Delta is candidate minus incumbent MAE, so negative favors candidate. Seven-day primary intervals:

| Recipe | Delta MAE | 95% interval |
|---|---:|---|
| E08 | -0.024724 | [-0.032871, -0.017231] |
| E13 | -0.027303 | [-0.034550, -0.020700] |
| E14 | -0.028474 | [-0.035744, -0.021751] |
| P13 | -0.033199 | [-0.044167, -0.023385] |

P13 one-day interval [-0.041033, -0.025345]; fourteen-day [-0.043855, -0.023731]. All four combination intervals exclude zero at every specified length; all other single candidates' intervals favor I, including L. These are descriptive paired intervals, not familywise-adjusted hypothesis tests or independent confirmation. See [all 45 bootstrap records](../../results/C015/final/bootstrap.csv).

### Distributional forecasting

P13 centers the residual distribution on E08 with member spread as the second feature. Each lead has a depth-two greedy tree trained on prior-year residuals; each leaf issues 99 equiprobable inverse-empirical-CDF residual supports. The independent auditor reconstructed the support, membership and scores and checked the issued median against the point export.

Pooled CRPS: **0.931423°C**. Nine pinball scores and lead/month/season diagnostics are archived.

| Nominal central interval | Empirical coverage | Mean width (°C) | Interval score |
|---:|---:|---:|---:|
| 50% | 50.95% | 2.12336 | 4.18881 |
| 80% | 81.45% | 4.25122 | 6.01772 |
| 90% | 91.43% | 5.76656 | 7.33829 |
| 95% | 95.30% | 6.91074 | 8.59606 |

The pooled intervals are slightly conservative. Pooled calibration does not imply conditional calibration at every lead, season or weather regime.

### Compute and seed sensitivity

Median annual fit+prediction seconds, incremental combinations versus inclusive serial base-model cost:

| Recipe | Incremental | Inclusive |
|---|---:|---:|
| E08 | 0.049500 | 189.645219 |
| E13 | 0.063721 | 204.967990 |
| E14 | 0.123430 | 205.055597 |
| P13 | 3.677538 | 192.967226 |

The incumbent is the best individual model; L and R trade modest accuracy loss for lower measured compute. G, K, A and H provide additional speed/accuracy trade-offs. S, N and X are dominated on the primary single-model MAE/compute frontier. All four combinations lie on their own incremental frontier. Inclusive timings sum the required independently timed base fits and combination stage, not actual end-to-end latency; machine variability, loading and CI orchestration limit interpretation. Peak RSS is archived per run, including process overhead; do not sum peaks as a memory estimate.

| Model | Seed42 MAE | Seed123 MAE | Seed2026 MAE | Range |
|---|---:|---:|---:|---:|
| M | 1.374273 | 1.362188 | 1.359274 | 0.014999 |
| S | 1.404680 | 1.413311 | 1.404544 | 0.008767 |
| N | 1.415439 | 1.412960 | 1.424496 | 0.011536 |
| W | 1.493635 | 1.516115 | 1.508853 | 0.022481 |

No alternative seed displaces the incumbent. The selected seed42 ranking remains unchanged.

## Reproducibility and evidence

- [Frozen cohort audit](https://github.com/sguzman/aevral/actions/runs/46094708880) and [preflight](https://github.com/sguzman/aevral/actions/runs/46095137641) passed.
- [80-fit run](https://github.com/sguzman/aevral/actions/runs/46096402389) and [16-combination run](https://github.com/sguzman/aevral/actions/runs/46097863362) both succeeded. All 96 outcome records are measured, independently audited.
- [Final re-ingestion, aggregation and archival](https://github.com/sguzman/aevral/actions/runs/46098369289) passed, including checksum verification before and after copying numerical evidence to Git and remote readback. Final archive artifact 15244376508 (90-day retention); per-origin artifacts have finite retention. The 343-file [Git archive](../../results/C015/final/) is durable and contains 342 hashed files plus SHA256SUMS.
- [Fit registry](../../results/C015/final/fits.csv), [combination registry](../../results/C015/final/combinations.csv), [annual scores](../../results/C015/final/annual.csv), [pooled scores](../../results/C015/final/pooled.csv), [rankings](../../results/C015/final/ranking.csv), [lead/month/season diagnostics](../../results/C015/final/diagnostics.csv), [distributional diagnostics](../../results/C015/final/probabilistic-pooled.csv), [coverage](../../results/C015/final/coverage.csv), [costs](../../results/C015/final/combination-costs.csv), [seed sensitivity](../../results/C015/final/seed-sensitivity.csv), [source and aggregate digests](../../results/C015/final/SHA256SUMS). Raw snapshots remain in the frozen cache; reconstructing the runs requires the identical inputs and pinned build environment, not a new upstream download assumed identical.
- In a checkout, verify the durable archive with `cd results/C015/final && sha256sum -c SHA256SUMS`. The three workflow launch files record the source runs; scripts document the exact ingestion and evaluation commands. Reproduction must not overwrite the completed frozen archive.

## Program conclusion and limitations

C003–C015 are complete. Earlier resource blocks in C004 (two finalists) and C012 (three N-HiTS evaluations) remain explicitly documented, not converted into measurements. C013 and C014 completed their 48-evaluation budgets each. The final shortlist is a finite comparison, not an exhaustive enumeration of all algorithms, sites or covariates. No external meteorological forecast inputs, other stations, new observations or prospective evaluation were authorized. A genuinely new prospective test would require separate approval and a frozen recipe, without tuning on its outcomes. Stop here: the authorized program is complete.
