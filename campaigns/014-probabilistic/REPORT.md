# C014 — Probabilistic forecasting results

## Status

Complete: all 48/48 frozen recipe-year outcomes measured and audited. No new base-model fits or observations. Best distributional CRPS: P13 (0.97156891°C), with point median MAE 1.34945390°C. P11 has the lowest 80% and 90% interval score, P13 the lowest 50% and 95% interval score. These are descriptive development results, not an independent holdout or a claim of universal calibration.

## Design and provenance

See [manifest](manifest.md) and [export format](EXPORT.md). Same 2021 and 2022 assessment cohort as C013: 7,501 and 7,639 complete 24-hour origins, totaling 363,360 target rows. Fit using only prior-fold predictions and residuals: 2020 for 2021; 2020–2021 for 2022. I = incumbent, B = C013 E08 equal mean of I,L,R,M. B was selected using these development years in C013, so even temporally separated calibrations do not remove model-selection exposure. No 2023–26 assessment labels used in C014.

Finite equal-mass support CRPS is exact for the issued distribution, not an analytic continuous Gaussian or Student-t score. Gaussian/t heads are residual postprocessors. Conformal outputs are interval-only; their point center is a median proxy; no CRPS/pinball ranking or iid exchangeability guarantee is claimed. All metrics are Celsius (coverage dimensionless), pooled using scalar counts, RMSE using squared errors. Pinball is the arithmetic average across nine frozen quantile levels.

## Full pooled ranking

Distribution recipes sorted by CRPS, then interval-only recipes in ID order. Interval columns refer to nominal 90% intervals.

| ID | Recipe | MAE | CRPS | Mean pinball | Coverage | Width | Interval score |
|---|---|---:|---:|---:|---:|---:|---:|
| P13 | Distributional tree B | 1.34945 | 0.97157 | 0.33432 | 89.86% | 5.69425 | 7.69857 |
| P20 | Member-specific mixture | 1.35132 | 0.97305 | 0.33519 | 91.27% | 5.90999 | 7.74098 |
| P19 | Pooled residual mixture | 1.35141 | 0.97308 | 0.33519 | 91.29% | 5.91158 | 7.74036 |
| P07 | Conditional Gaussian B | 1.35167 | 0.97417 | 0.33563 | 90.67% | 5.80792 | 7.74008 |
| P04 | Lead residual B | 1.35153 | 0.97538 | 0.33650 | 90.01% | 5.68541 | 7.79984 |
| P11 | Quantile regression B | 1.35162 | 0.97556 | 0.33652 | 89.15% | 5.52806 | 7.69448 |
| P06 | Gaussian B | 1.35136 | 0.97573 | 0.33682 | 90.73% | 5.83115 | 7.80211 |
| P15 | Center analog B | 1.35807 | 0.97925 | 0.33813 | 89.82% | 5.83506 | 7.85482 |
| P16 | Spread analog B | 1.35829 | 0.97984 | 0.33828 | 89.84% | 5.80961 | 7.85632 |
| P09 | Conditional Student-t B | 1.35167 | 0.98814 | 0.34144 | 85.26% | 4.87818 | 7.92212 |
| P08 | Student-t B | 1.35136 | 0.98878 | 0.34224 | 85.45% | 4.89769 | 7.97340 |
| P02 | Global residual B | 1.35121 | 0.98889 | 0.34498 | 90.00% | 5.87446 | 8.20515 |
| P12 | Distributional tree I | 1.38234 | 0.99263 | 0.34073 | 89.82% | 5.78610 | 7.80517 |
| P10 | Quantile regression I | 1.38167 | 0.99472 | 0.34231 | 89.06% | 5.61069 | 7.78100 |
| P03 | Lead residual I | 1.38140 | 0.99578 | 0.34303 | 89.91% | 5.79509 | 7.92396 |
| P05 | Gaussian I | 1.38149 | 0.99605 | 0.34327 | 90.59% | 5.92561 | 7.92148 |
| P14 | Center analog I | 1.39179 | 1.00127 | 0.34488 | 89.74% | 5.92532 | 7.96672 |
| P01 | Global residual I | 1.37998 | 1.00771 | 0.35088 | 89.89% | 5.95758 | 8.31332 |
| P18 | Raw eight-member | 1.38998 | 1.16647 | 0.49202 | 38.54% | 1.74520 | 16.33157 |
| P17 | Raw four-member | 1.36188 | 1.18886 | 0.54138 | 24.17% | 0.99991 | 19.55078 |
| P21 | Symmetric conformal I | 1.37862 | — | — | 89.04% | 5.61131 | 7.92698 |
| P22 | Symmetric conformal B | 1.35048 | — | — | 89.14% | 5.50646 | 7.80660 |
| P23 | Asymmetric conformal B | 1.35048 | — | — | 89.10% | 5.49984 | 7.80798 |
| P24 | Spread-normalized conformal B | 1.35048 | — | — | 89.57% | 7.48909 | 9.40051 |

## Calibration and sharpness

| Nominal | P13 coverage | P13 width | P13 interval score | Best interval-score recipe |
|---|---:|---:|---:|---|
| 50% | 48.39% | 2.07591 | 4.36447 | P13 |
| 80% | 79.16% | 4.19307 | 6.30094 | P11 (6.29047) |
| 90% | 89.86% | 5.69425 | 7.69857 | P11 (7.69448) |
| 95% | 94.13% | 6.80891 | 9.11993 | P13 |

P13 is the best full distribution by CRPS and mean pinball (0.33431844°C). P20 and P19 are close (0.97305266 and 0.97307595°C). The 90% interval-score winner P11 has 89.15% coverage; its 95% interval is identical to its 90% interval because the quantile grid clamps at 0.05 and 0.95. This is an important tail limitation, not evidence that the intervals achieve 95% coverage.

Raw ensembles P17 and P18 cover only 24.17% and 38.54% at nominal 90%. Adding calibrated residuals (P19/P20) yields about 91.3%, much better distribution scores and sensible widths. Raw member disagreement is not a sufficient uncertainty estimator here. Variance-matched Student-t (P08/P09) undercovers at 90% (~85%); heavier tails do not automatically improve finite-grid central interval calibration. Spread-normalized conformal P24 has wide 90% bands (7.48909°C) and worse interval score (9.40051) than P22 (5.50646°C, 7.80660), without hitting nominal coverage. P22 is the best conformal recipe at 90% by interval score. All variants and all four levels retained in pooled diagnostics.

## Lead and month diagnostics

| Horizon | P13 median MAE | CRPS | 90% coverage | Width |
|---:|---:|---:|---:|---:|
| 1h | 0.54473 | 0.39638 | 90.83% | 2.41816 |
| 6h | 1.12731 | 0.81459 | 89.62% | 4.76937 |
| 12h | 1.41986 | 1.02162 | 89.58% | 6.01228 |
| 18h | 1.60283 | 1.15163 | 89.79% | 6.70312 |
| 24h | 1.73638 | 1.24908 | 89.66% | 7.20873 |

June is P13's hardest month (CRPS 1.23026°C, 84.62% 90%-interval coverage); August has 94.54% coverage. Pooled near-nominal coverage therefore masks important conditional miscalibration. All 24 leads and 12 months for all recipes are retained. No claim of conditional validity or statistical significance for overlapping windows.

## Costs and numerical behavior

P13 incremental fit times 0.090154s/0.223137s, issue+score+compression 0.839651s/0.863619s for 2021/22. P20 fit 0.028255s/0.075113s, issue+score+compression 1.123036s/1.272998s. These are cached-base-forecast postprocessing timings, not live end-to-end inference. Base-model fit/predict times in [C013 source inventory](../013-ensembles/source-inventory.csv). Total scorer 22.11s/25.17s, peak RSS 36,932/69,964 KiB; independent auditor 8.22s/9.71s, peak RSS 67,988/84,448 KiB. Quantile rearrangement occurred for P10 (3,399/2,500 rows) and P11 (3,941/3,824 rows) before issuing noncrossing supports. No failed assessment outcome.

## Reproducibility and evidence

- [Pooled diagnostics](../../results/C014/pooled-diagnostics.csv): 888 rows, 24 recipes × (overall + 24 leads + 12 months).
- [2021](../../results/C014/2021/) and [2022](../../results/C014/2022/): 888 metric rows each, overall/costs, 24 exact fitted parameter tables per year, original assessment hashes and audit records.
- [Assessment CI](https://github.com/sguzman/aevral/actions/runs/46091343709): independently reconstructs 8,720,640 forecasts (48 recipe-years), verifies all 1,776 metric rows, scores with independent CRPS identity. Parameter structure validated; not independent refitting of every estimator.
- [Archival CI](https://github.com/sguzman/aevral/actions/runs/46091626977): verifies 106 original file hashes.
- [Final integrity CI](https://github.com/sguzman/aevral/actions/runs/46093205838): verifies all 52 durable metadata checksums, overall subsets and byte-exact independent reproduction of pooled diagnostics.
- Source joined artifact 15239944097; assessment artifacts 15241032976 and 15241106780 (finite retention). Durable aggregate/parameter evidence and hashes in Git; raw per-record forecast artifacts not permanent. No joint 24-hour trajectory distribution or fresh holdout.

C014 closes after all frozen outcomes and durable checks; C015 is the final retrospective tournament, not another distributional search.
