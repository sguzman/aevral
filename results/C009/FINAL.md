# C009 — Final neural architecture ranking

All 42 screens and 374 additional annual evaluations measured, zero blocked. Exactly 34 promoted recipes (two per each of 17 families), 408 finalist outcomes across four annual folds and three seeds; eight nonpromoted screens retained. All four annual CI batches and artifact names reconciled. Each finalist has 30,138 matched origins per seed, 90,414 origin-seed pairs (not independent observations). 162 tests passed per annual job. Raw prediction payloads have not been independently downloaded/rescored; archived CI aggregate/diagnostic logs and artifact metadata are the evidence.

## Pooled development ranking

MAE weighted by origins across years and seeds; RMSE is square root of weighted squared logged RMSE. Each seed pooled across years; each fold pooled across seeds. Seed SD is population SD of the three pooled seed MAEs, descriptive only. Rounded source metrics imply rounding-level precision; no iid inference. Ranking by pooled MAE then RMSE; no exact ties. [Full CSV](pooled.csv), [registry](registry.csv).

| Rank | Method | Recipe | Pooled MAE °C | RMSE °C | Seed SD °C |
|---:|---|---|---:|---:|---:|
| 1 | N10 | nhits-w64 | 1.45889978 | 1.94617663 | 0.00878335 |
| 2 | N09 | nbeats-w64-b4 | 1.46904639 | 1.95497396 | 0.01903910 |
| 3 | N09 | nbeats-w64-b2 | 1.46927477 | 1.95888874 | 0.02156797 |
| 4 | N10 | nhits-w128 | 1.47727896 | 1.96949304 | 0.01482042 |
| 5 | N04 | gru-p3-h16 | 1.55925402 | 2.06809526 | 0.01304146 |
| 6 | N05 | lstm-p6-h16 | 1.56115040 | 2.06740365 | 0.01276298 |
| 7 | N05 | lstm-p3-h16 | 1.56215561 | 2.07071758 | 0.01359425 |
| 8 | N13 | tide-w64 | 1.56844384 | 2.09509419 | 0.03720694 |
| 9 | N04 | gru-p1-h16 | 1.56989545 | 2.07814244 | 0.01490410 |
| 10 | N17 | moe-e4-w32 | 1.58093231 | 2.10431967 | 0.01461967 |
| 11 | N13 | tide-w128 | 1.59177629 | 2.12704693 | 0.04364563 |
| 12 | N17 | moe-e4-w64 | 1.59202542 | 2.11543585 | 0.01197398 |
| 13 | N03 | rnn-p6-h16 | 1.60152260 | 2.11688025 | 0.01983554 |
| 14 | N14 | patchtst-p6-d16 | 1.60552985 | 2.12210261 | 0.03464539 |
| 15 | N15 | attention-p6-d8 | 1.60660348 | 2.12203786 | 0.02930741 |
| 16 | N03 | rnn-p3-h16 | 1.60975501 | 2.12617807 | 0.01811150 |
| 17 | N15 | attention-p3-d8 | 1.61002348 | 2.12494172 | 0.02791313 |
| 18 | N14 | patchtst-p3-d16 | 1.61094893 | 2.12699320 | 0.03198009 |
| 19 | N07 | tcn-p3-c8 | 1.62748234 | 2.14863119 | 0.04081677 |
| 20 | N06 | cnn-p3-c8 | 1.62751146 | 2.15071630 | 0.04941477 |
| 21 | N02 | resmlp-w64-b2 | 1.62761115 | 2.15688123 | 0.00722253 |
| 22 | N01 | mlp-w128 | 1.63748822 | 2.17807262 | 0.03339774 |
| 23 | N07 | tcn-p6-c8 | 1.64648531 | 2.16861988 | 0.03730681 |
| 24 | N02 | resmlp-w64-b4 | 1.65528495 | 2.18533914 | 0.01190634 |
| 25 | N06 | cnn-p6-c8 | 1.65540018 | 2.18158493 | 0.04925249 |
| 26 | N01 | mlp-w256 | 1.66247194 | 2.20402048 | 0.05550720 |
| 27 | N11 | dlinear-k7 | 1.68312046 | 2.23049733 | 0.04449015 |
| 28 | N12 | nlinear-direct | 1.68493764 | 2.23723293 | 0.00902679 |
| 29 | N12 | nlinear-lead-bias | 1.68517409 | 2.23767111 | 0.00905490 |
| 30 | N11 | dlinear-k25 | 1.71079100 | 2.26662637 | 0.04820239 |
| 31 | N16 | ssm-h16 | 1.72207967 | 2.26989464 | 0.01401528 |
| 32 | N16 | ssm-h32 | 1.72232335 | 2.27030505 | 0.01335386 |
| 33 | N08 | seq2seq-h32 | 1.73094831 | 2.29195149 | 0.05180685 |
| 34 | N08 | seq2seq-h16 | 1.77620741 | 2.36306142 | 0.04035012 |

## Decision

Best C009: nhits-w64, MAE 1.45889978°C, RMSE 1.94617663°C. C003 full-training MLP64 reference MAE 1.34781461°C, RMSE 1.80956416°C. Relative MAE difference 8.242% (C009 minus reference). C009 used 8,192 train origins, 12 epochs Adam .001 vs different full-training incumbent recipe; comparison is descriptive and not a controlled architecture-only comparison. No C009 finalist improves the full-training reference; retain incumbent. Do not infer architecture superiority from unequal training budgets. Development periods are historical, not independent holdouts. No further C009 recipes authorized. Proceed C010 forecast formulation and objective.

CI: [batch 1](https://github.com/sguzman/aevral/actions/runs/46038328288), [batch 2](https://github.com/sguzman/aevral/actions/runs/46040581126), [batch 3](https://github.com/sguzman/aevral/actions/runs/46049602720), [batch 4](https://github.com/sguzman/aevral/actions/runs/46051292157).
