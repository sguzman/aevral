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
