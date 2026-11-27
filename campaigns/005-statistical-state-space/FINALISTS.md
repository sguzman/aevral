# C005 frozen annual finalists
2019 screen complete: 36/36 measured, 34 point + 2 variance-only. Selection by MAE then RMSE, no ties affecting within-family selection. Reuse 2019 scores and C003 incumbent; refit identical recipes on expanding training for 2020, 2021, 2022. No parameter selection based on later folds. Retain structurally equivalent models as explicitly predeclared family representatives, not independent evidence. Preserve capped optimizer flags. 
| Family | Point configurations |
|---|---|
| S01 | ar-72, ar-48 |
| S02 | arx-72, arx-48 |
| S03 | var-1 |
| S04 | varx-1 |
| S05 | arima-101, arima-211 |
| S06 | arimax-101, arimax-111 |
| S07 | sarima-101-010, sarima-101-100 |
| S08 | ets-aaa, ets-ann |
| S09 | harmonic-2-ar1, harmonic-1-ar1 |
| S10 | struct-season, struct-level |
| S11 | dlm-harmonic, dlm-trend-harmonic |
| S12 | switch-ar6, switch-ar1 |
Variance-only S13: both garch-ar24 and garch-ar6, compare separately by Gaussian NLL and coverage. Point means only for parity; exclude from point ranking. 22 point configurations plus two variance-only, 72 additional annual outcomes. Group jobs by fold and implementation module, <=6 parallel, <=30min/job; preserve all configs and code hashes, logs and artifacts. No paid compute, no retrospective 2023+ until C015. Pool MAE and RMSE with origin counts and squared RMSE, variance NLL and coverage by counts, variance calibration ratio requires pooled sums or report fold-specific ratios only. 
