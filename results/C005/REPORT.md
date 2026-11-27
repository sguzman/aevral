# C005 final report — classical statistical and state-space forecasting
Complete within the frozen manifest: 36 screens (34 point, two variance-only), 72 annual finalist outcomes; 108 unique measured outcomes. 22 point finalists plus C003 MLP64, two uncertainty finalists. No blocked outcomes. All 15 annual jobs passed, 76 numerical tests. 2019–2022 matched cohorts total 30138 origins, 723312 hourly targets per complete model. No 2023+ observations used for fitting or selection. 
## Pooled point forecasts
Weighted by annual origin count; RMSE pooled via weighted squared RMSE. Source: registry.csv and C003 incumbent annual rows (legacy excluded). Values computed from archived rounded scores, not independently recomputed from prediction artifacts.
| Rank | Model | MAE °C | RMSE °C |
|---:|---|---:|---:|
| 1 | mlp64 | 1.34781461 | 1.80956416 |
| 2 | arx-72 | 1.59560910 | 2.10987644 |
| 3 | arx-48 | 1.60803790 | 2.12496339 |
| 4 | ar-72 | 1.65126954 | 2.19941283 |
| 5 | ar-48 | 1.68415067 | 2.24138571 |
| 6 | struct-season | 1.75675440 | 2.33990946 |
| 7 | dlm-harmonic | 1.75675440 | 2.33990946 |
| 8 | sarima-101-010 | 1.82831081 | 2.45858092 |
| 9 | harmonic-2-ar1 | 1.87699045 | 2.49447100 |
| 10 | varx-1 | 1.88080063 | 2.51905938 |
| 11 | arimax-101 | 1.91108910 | 2.56102287 |
| 12 | harmonic-1-ar1 | 1.93404979 | 2.55685113 |
| 13 | dlm-trend-harmonic | 2.02586399 | 2.71780244 |
| 14 | ets-aaa | 2.21022651 | 2.89016104 |
| 15 | sarima-101-100 | 2.21761437 | 2.94497815 |
| 16 | arimax-111 | 2.31308530 | 3.04695710 |
| 17 | var-1 | 2.78300000 | 3.75694117 |
| 18 | arima-101 | 2.95061010 | 3.96200854 |
| 19 | struct-level | 2.97289551 | 3.98158895 |
| 20 | ets-ann | 2.98426450 | 4.01674870 |
| 21 | switch-ar1 | 3.28442068 | 4.42217887 |
| 22 | switch-ar6 | 3.30402486 | 4.42191499 |
| 23 | arima-211 | 3.45590241 | 4.70156854 |
Best statistical model arx-72; MAE 1.59560910, 18.38% above incumbent. No candidate beats MLP64 pooled. Equivalent structural seasonal and DLM harmonic results are not independent model evidence. 
## Conditional uncertainty (not point-ranking candidates)
| Model | Pooled Gaussian NLL | Empirical nominal 90% coverage |
|---|---:|---:|
| garch-ar24 | 2.20079363 | 89.2215% |
| garch-ar6 | 2.76740362 | 90.7231% |
Coverage and NLL pooled by origin counts; calibration ratios are reported by fold only in variance-pooled.csv. Gaussian approximation, no exact distribution claim. GARCH means reuse corresponding AR recipes; 2019 mean parity confirmed, annual AR6/24 not independently screened in S01, so no claim of independent annual parity. 
## Audit and limitations
Verified registry keys, status, exact per-fold cohorts, 23 four-fold point models including incumbent, 8 uncertainty records, 44 archived job logs matching 44 distinct job IDs. Ingestion checked 24 lead scores per measured outcome and 24 variance leads per GARCH outcome. Logs preserve fit times, parameter counts, RSS, coefficients, optimizer flags and month metrics; no controlled hardware runtime benchmark or independent re-score of every prediction. CI artifacts expire after 30 days. Capped conditional optimizer/EM results retained as approximate; fixed-window filtering, bounded train-only grids and conditional fitting as specified, not full-series ML. No extra per-origin history or future covariates. 
Annual CI: https://github.com/sguzman/aevral/actions/runs/45914280432 . Individual job references and code SHAs in registry. All 13 families resolved. No additional fitting required for C005. Proceed to C006.
