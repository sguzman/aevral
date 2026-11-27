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

## F05–F07 target transformations and correction

6/6 measured, zero blocked, 167 tests passed per job. All six artifact names reconciled. Same frozen cohort and training protocol; correction base fit uses purged chronological subset and separate later correction fit. Raw prediction payloads not independently rescored.

| Recipe | MAE °C | RMSE °C |
|---|---:|---:|
| correction-h16 | 1.44854255 | 1.89339741 |
| daily-residual-mlp64 | 1.47634895 | 1.93194096 |
| daily-residual-nhits64 | 1.47790090 | 1.94161721 |
| difference-nhits64 | 1.54377350 | 2.05825357 |
| difference-mlp64 | 1.64026118 | 2.12417854 |
| correction-affine | 1.80854257 | 2.36242434 |

[CI](https://github.com/sguzman/aevral/actions/runs/46058395640). Total 14/28 screens measured. Best so far joint-nhits64 MAE 1.42065820. Next F08–F10.

## F08–F10 objective screen checkpoint

6/6 archived, 20/28 total measured, zero blocked among archived jobs. 171 tests passed per archived job. [CI](https://github.com/sguzman/aevral/actions/runs/46059749684); raw predictions not independently rescored.

| Recipe | MAE °C | RMSE °C |
|---|---:|---:|
| mae-nhits64 | 1.41301012 | 1.88104693 |
| huber-nhits64 | 1.41552914 | 1.88252049 |
| huber-mlp64 | 1.49535192 | 1.95522636 |
| pinball-multi | 1.50215236 | 1.94332630 |
| mae-mlp64 | 1.50348015 | 1.96314114 |
| pinball-median | 1.52824187 | 1.98736169 |
Multi-quantile crossing leads: 268/178920 (unrearranged).

## F11–F14 regularization and constraints checkpoint

6/8 archived, 26/28 total measured, zero blocked among archived jobs. 173 tests passed per archived job. [CI](https://github.com/sguzman/aevral/actions/runs/46060046667); raw predictions not independently rescored.

| Recipe | MAE °C | RMSE °C |
|---|---:|---:|
| lead-early | 1.47598835 | 1.93039143 |
| trajectory-slope | 1.49719943 | 1.95970325 |
| multitask-mean-change | 1.50042891 | 1.95906824 |
| bound-clamp | 1.52449139 | 1.98742079 |
| multitask-mean | 1.53441585 | 1.99695425 |
| lead-late | 1.55850193 | 2.02829224 |
- trajectory-slope: slope MAE/curvature MAE/counts 0.60324108/0.86938682/171465/164010.
- bound-clamp: outside/clipped/total 0/0/178920.
