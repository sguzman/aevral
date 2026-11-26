# C005 statistical/state-space results
S01–S04 [CI](https://github.com/sguzman/aevral/actions/runs/45900349024), code efa95b5624c26666fb3e9cbf2dbb4b2e16a0e3ff, 57 numerical tests per completed job. 10/36 screening outcomes archived. All models use the fixed cohort; recursive forecasts receive no future weather observations. ARX uses origin weather frozen; VAR fits adjacent complete training origin states. Impulse growth is a finite diagnostic, not exact spectral radius.

| Method | Configuration | MAE °C | RMSE °C |
|---|---|---:|---:|
| S01 | ar-6 | 3.27675238 | 4.37092148 |
| S01 | ar-24 | 1.64809934 | 2.18433744 |
| S01 | ar-48 | 1.58880760 | 2.09521027 |
| S01 | ar-72 | 1.54626475 | 2.04673550 |
| S03 | var-1 | 2.77049148 | 3.73547321 |
| S04 | varx-1 | 1.79554219 | 2.41678868 |

| S02 | arx-6 | 1.90056213 | 2.60701203 |
| S02 | arx-24 | 1.58645938 | 2.08907827 |
| S02 | arx-48 | 1.54393141 | 2.03388679 |
| S02 | arx-72 | 1.53087031 | 2.01724040 |

All S01–S04 jobs passed. Best ARX72 MAE 1.53087031; best AR72 1.54626475. Promote AR72/48, ARX72/48, VAR1 and VARX1.

## S05–S07 conditional ARIMA (batch 1)
[CI](https://github.com/sguzman/aevral/actions/runs/45900629055), code 58fbc0f94d4df47b34a8d963f47cef0d6b2d6091. All nine jobs completed successfully; first six results archived here, remaining three pending ingestion. 62 tests passed. Conditional optimizer uses 2048 train-only windows; finite capped fits are approximate, not exact ML.

| Model | MAE | RMSE | Optimizer converged |
|---|---:|---:|---|
| arimax-101 | 1.81305519 | 2.46355131 | true |
| arimax-111 | 2.07347384 | 2.71867701 | false |
| arima-101 | 2.89296435 | 3.84514565 | true |
| sarima-111-100 | 2.74570304 | 3.67784674 | true |
| arima-111 | 3.44694169 | 4.66211547 | true |
| sarima-101-010 | 1.70594491 | 2.27617711 | true |

### Remaining ARIMA measurements
| Model | MAE | RMSE | Optimizer converged |
|---|---:|---:|---|
| arima-211 | 3.42059433 | 4.62695903 | true |
| arima-201 | 3.82301171 | 5.06843975 | true |
| sarima-101-100 | 2.13723251 | 2.82288611 | true |

All nine ARIMA outcomes archived. 19/36 screens measured. Provisional finalists: S05: arima-101, arima-211; S06: arimax-101, arimax-111; S07: sarima-101-010, sarima-101-100. Preserve capped-fit flags in comparisons.

## S08–S09 smoothing and harmonic regression (first batch)
[CI](https://github.com/sguzman/aevral/actions/runs/45901037832), code 1f1bd8c09d360fca15948dc794774ff9e291bb0a. All eight jobs succeeded. First five archived; remaining three pending ingestion.

| Model | MAE | RMSE |
|---|---:|---:|
| harmonic-2-ar1 | 1.77158598 | 2.35050457 |
| ets-aan | 7.33301973 | 10.16491189 |
| ets-ann | 2.93355531 | 3.91434094 |
| harmonic-2-ar6 | 1.88417564 | 2.54701774 |
| ets-aaa | 2.06852752 | 2.71455735 |

### Remaining smoothing and harmonic measurements
| Model | MAE | RMSE |
|---|---:|---:|
| ets-aadn | 3.44283257 | 4.64128599 |
| harmonic-1-ar1 | 1.81928833 | 2.39677695 |
| harmonic-1-ar6 | 1.88142584 | 2.51775159 |

All eight S08–S09 outcomes archived; 27/36 measured. Provisional finalists: S08: ets-aaa, ets-ann; S09: harmonic-2-ar1, harmonic-1-ar1.

## S10–S11 structural/Kalman measurements
[CI](https://github.com/sguzman/aevral/actions/runs/45913783874), code a08a2cf6bf26c64058457a0217f2a1a3091c3c64. 4/5 archived; 72 numerical tests passed in completed jobs.

| Model | MAE | RMSE |
|---|---:|---:|
| struct-season | 1.67113899 | 2.20514152 |
| struct-level | 2.91876103 | 3.87648582 |
| dlm-harmonic | 1.67113899 | 2.20514152 |
| struct-trend | 7.31826143 | 10.09863143 |

### Final Kalman outcome
| Model | MAE | RMSE |
|---|---:|---:|
| dlm-trend-harmonic | 1.92420168 | 2.55612939 |

All five S10–S11 jobs succeeded. 32/36 screening outcomes archived. Structural seasonal and dynamic harmonic models have identical scores, consistent with rotation-equivalent state parameterizations and isotropic seasonal process noise; not independent evidence. Provisional finalists: {"S10":["struct-season","struct-level"],"S11":["dlm-harmonic","dlm-trend-harmonic"]}.
