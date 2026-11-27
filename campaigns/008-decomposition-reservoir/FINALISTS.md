# C008 frozen annual finalists

Promoted top two per family from the 36 measured 2019 screens, ranked by MAE, RMSE, then measured fit+prediction seconds. No recipes changed. Symbolic depth1 and depth2 eight-term recipes tied on rounded MAE/RMSE; fit+prediction seconds {"symbolic-depth1-t8":0.848793,"symbolic-depth2-t8":1.838305}. Both promoted. Reuse C003 MLP64 and each finalist's original 2019 score (seed42 for stochastic models). Deterministic finalists: three additional folds. ESN/ELM: seeds 42,123,2026 across four folds, excluding existing 2019 seed42. 86 additional outcomes, exact frozen matrix below.

| Family | Configuration | 2019 MAE °C | 2019 RMSE °C |
|---|---|---:|---:|
| D01 | fourier-k6-l001 | 1.47326621 | 1.94231164 |
| D01 | fourier-k6-l01 | 1.47935499 | 1.94611855 |
| D02 | haar-m32-l001 | 1.47457748 | 1.94497482 |
| D02 | haar-m32-l01 | 1.47679777 | 1.94572315 |
| D03 | ssa-l24-r8 | 1.73745553 | 2.27052059 |
| D03 | ssa-l36-r8 | 1.79963437 | 2.35929816 |
| D04 | dmd-l24-r8 | 1.63639364 | 2.15934098 |
| D04 | dmd-l24-r4 | 1.81813557 | 2.34990010 |
| D05 | edmd-l12-l001 | 2.05963056 | 2.71789011 |
| D05 | edmd-l12-l01 | 2.16448064 | 2.85948682 |
| D06 | esn-m128-leak03 | 1.57284555 | 2.05669316 |
| D06 | esn-m64-leak03 | 1.58162346 | 2.06914311 |
| D07 | elm-m128-l001 | 1.48624269 | 1.94727113 |
| D07 | elm-m128-l01 | 1.49773178 | 1.95946007 |
| D08 | markov-b16-hour | 1.95897401 | 2.57619151 |
| D08 | markov-b8-hour | 2.26351402 | 2.97072960 |
| D09 | symbolic-depth1-t8 | 1.59234152 | 2.06505401 |
| D09 | symbolic-depth2-t8 | 1.59234152 | 2.06505401 |

[Exact annual matrix](finalist-matrix.csv). No fitting performed in this commit.
