# C009 screen results

Frozen 2019 cohort: 7,455 origins, seed42, 8,192 training origins, 12 epochs Adam .001. C003 full-training MLP64 2019 MAE 1.32617622 (different training budget). 42 of 42 screened (42 measured, 0 blocked). Rust and historical architecture regression CI passed. See final ranking below.

| Recipe | MAE °C | RMSE °C | Parameters | Fit seconds | Predict seconds | Clipped updates |
|---|---:|---:|---:|---:|---:|---:|
| mlp-w128 | 1.52309663 | 1.97813457 | 13464 | 12.143077 | 0.110769 | 0 |
| resmlp-w64-b2 | 1.52771089 | 1.98283934 | 23384 | 9.932843 | 0.070316 | 817 |
| resmlp-w64-b4 | 1.53330066 | 2.00639008 | 40024 | 27.104756 | 0.218524 | 1249 |
| mlp-w256 | 1.53909188 | 1.99918547 | 26904 | 13.398324 | 0.128007 | 0 |
| mlp-128x64 | 1.59058790 | 2.06952853 | 20184 | 19.142084 | 0.166310 | 0 |
| resmlp-w128-b2 | 1.62040418 | 2.09164529 | 79512 | 34.728919 | 0.243631 | 1130 |

[CI](https://github.com/sguzman/aevral/actions/runs/46035179411) · [Registry](registry.csv).

## N03–N05 recurrent screens

| Recipe | MAE °C | RMSE °C | Status |
|---|---:|---:|---|
| gru-p3-h16 | 1.46936104 | 1.94220344 | measured |
| gru-p1-h16 | 1.49104802 | 1.96415019 | measured |
| lstm-p6-h16 | 1.49294941 | 1.96106871 | measured |
| gru-p6-h16 | 1.49626522 | 1.96546328 | measured |
| lstm-p3-h16 | 1.49842780 | 1.96858800 | measured |
| lstm-p1-h16 | 1.50672633 | 1.98627517 | measured |
| rnn-p6-h16 | 1.52063853 | 1.99555269 | measured |
| rnn-p3-h16 | 1.53504693 | 2.01174815 | measured |
| rnn-p1-h16 | 1.63448142 | 2.14369256 | measured |

[Recurrent CI](https://github.com/sguzman/aevral/actions/runs/46035599511), 152 tests passed; legacy regression and Rust CI passed.

## N06–N08 temporal and encoder-decoder screens

| Recipe | MAE °C | RMSE °C | Status |
|---|---:|---:|---|
| tcn-p3-c8 | 1.50925824 | 1.98407349 | measured |
| cnn-p6-c8 | 1.51056969 | 1.98513181 | measured |
| cnn-p3-c8 | 1.51923450 | 1.99192790 | measured |
| tcn-p6-c8 | 1.52234152 | 1.99648094 | measured |
| seq2seq-h16 | 1.78546137 | 2.35692147 | measured |
| seq2seq-h32 | 1.84231475 | 2.38104296 | measured |

[CI](https://github.com/sguzman/aevral/actions/runs/46036072293), 155 tests; legacy regression and Rust CI passed.

## N09–N13 structured neural screens

| Recipe | MAE °C | RMSE °C | Status |
|---|---:|---:|---|
| nbeats-w64-b4 | 1.41791905 | 1.87640895 | measured |
| nhits-w64 | 1.42065820 | 1.88590767 | measured |
| nbeats-w64-b2 | 1.42066476 | 1.88201892 | measured |
| nbeats-w128-b2 | 1.44135567 | 1.90876908 | measured |
| nhits-w128 | 1.44186891 | 1.90523601 | measured |
| tide-w64 | 1.47447075 | 1.92984148 | measured |
| tide-w128 | 1.48508979 | 1.93877664 | measured |
| dlinear-k7 | 1.54565747 | 2.03441421 | measured |
| nlinear-lead-bias | 1.55929609 | 2.05906489 | measured |
| nlinear-direct | 1.55965220 | 2.05920005 | measured |
| dlinear-k25 | 1.57713214 | 2.07967357 | measured |

[CI](https://github.com/sguzman/aevral/actions/runs/46036545583), 157 tests; Rust CI passed.

## N14–N15 attention screens

| Recipe | MAE °C | RMSE °C | Status |
|---|---:|---:|---|
| patchtst-p3-d16 | 1.49223889 | 1.96051712 | measured |
| attention-p3-d8 | 1.49239237 | 1.95911318 | measured |
| attention-p6-d8 | 1.49426442 | 1.96565466 | measured |
| attention-p6-d16 | 1.50052953 | 1.97298473 | measured |
| patchtst-p6-d16 | 1.50811242 | 1.98090232 | measured |

[CI](https://github.com/sguzman/aevral/actions/runs/46037501450), 160 tests; Rust CI passed.

## N16–N17 state space and mixture screens

| Recipe | MAE °C | RMSE °C | Status |
|---|---:|---:|---|
| moe-e4-w32 | 1.48051782 | 1.94466437 | measured |
| moe-e4-w64 | 1.50044704 | 1.96768688 | measured |
| moe-e2-w64 | 1.51794632 | 1.98037224 | measured |
| ssm-h32 | 1.60270694 | 2.11159484 | measured |
| ssm-h16 | 1.60298236 | 2.11197678 | measured |

[CI](https://github.com/sguzman/aevral/actions/runs/46037920375), 162 tests; Rust CI passed. All 42 frozen screening recipes reconciled against matrix. Annual evaluation complete.

## Frozen promotion and annual evaluation

34 finalists, two per family, chosen by 2019 MAE/RMSE (no exact ties; timing tie-break not needed). Eight screened configurations excluded. See [promotion.csv](promotion.csv). Reuse the 34 seed42 2019 scores. Frozen 374 further evaluations in four sequential batches (96,96,96,86), each at most six concurrent fitting jobs. Ordering is fold, seed, family, within-family rank; no new model recipes. First batch triggered by workflow commit; subsequent batches dispatch only after previous batch is archived. All 42 screening records remain in registry. Full annual completion requires 408 finalist outcomes including reused screens, or explicit blocks. Annual evaluation complete.

### Annual batch 1 complete

[CI 46038328288](https://github.com/sguzman/aevral/actions/runs/46038328288): 96 measured, zero blocked, all 96 unexpired artifact names matched to frozen matrix; individual predictions validated in CI, not independently rescored. Archive checkpoint 2315f0f93b6a2536ed971461c546d83a89a434d3. Batch 2 launched via changing only workflow's default batch (dispatch connector unavailable), retaining sequential execution and identical training configuration.

### Annual batch 2 complete

[CI 46040581126](https://github.com/sguzman/aevral/actions/runs/46040581126): 96/96 measured, zero blocked; all 96 unexpired artifact names reconciled against frozen matrix. Registry contains 234 unique measured outcomes (42 screens + 192 additional); finalist coverage 226/408. Final archived job 140647062578, nhits-w128 2021 seed42: MAE 1.51547781°C, RMSE 2.01883986°C, 7,501 origins. 162 tests passed; predictions not independently rescored. Next frozen batch 3.

### Annual batch 3 complete

[CI 46049602720](https://github.com/sguzman/aevral/actions/runs/46049602720): all 96 measured, zero blocked; all 96 expected unexpired artifacts reconciled against frozen matrix. Registry: 330 unique measured outcomes, finalist coverage 322/408. Raw predictions not independently rescored. Final batch of 86 frozen evaluations selected next.

### Annual batch 4 complete; campaign closed

[CI](https://github.com/sguzman/aevral/actions/runs/46051292157): 86/86 measured, all 86 expected artifacts reconciled. All 416 unique outcomes archived. [Final report](FINAL.md) · [Pooled ranking](pooled.csv). Raw predictions not independently rescored.
