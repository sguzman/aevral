# C010 screening results

F01–F04: 8/28 measured, zero blocked. All eight artifacts matched to frozen recipes; 165 tests passed per job. 2019, seed42, 8192 training origins, 12 epochs Adam .001; matched 7455 evaluation origins. Raw prediction files not independently rescored. [CI](https://github.com/sguzman/aevral/actions/runs/46054536336).

| Method | Model | MAE °C | RMSE °C |
|---|---|---:|---:|
| F01 | joint-nhits64 | 1.42065820 | 1.88590767 |
| F02 | perlead-h16 | 1.51140087 | 1.98914805 |
| F01 | joint-mlp64 | 1.52449139 | 1.98742079 |
| F02 | perlead-h8 | 1.52598483 | 2.00802436 |
| F04 | dirrec-h16 | 1.92468459 | 2.63357226 |
| F04 | dirrec-h8 | 2.01778317 | 2.74242343 |
| F03 | recursive-h16 | 5.26813385 | 6.49916587 |
| F03 | recursive-h32 | 5.48816306 | 6.76080759 |

Screen only, not annual ranking. Next F05–F07. Recursive teacher-forcing/free-running mismatch is a candidate explanation for poor results, not a demonstrated causal finding.
