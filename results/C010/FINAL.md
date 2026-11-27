# C010 — Final forecast formulation and objective ranking

All 28 screens and 308 additional annual evaluations measured; zero blocked. All 28 recipes across 14 method groups promoted. All 336 outcomes reconciled with the frozen registry; all four annual batches reconciled with CI metadata. Each recipe has 30,138 matched origins per seed, 90,414 origin-seed pairs, not independent observations. 173 tests passed per additional annual job. Raw prediction payloads not independently downloaded/rescored; archived CI aggregates/diagnostics and artifact metadata are the evidence. Screen code/test counts differ by implementation stage as recorded in the registry and screening README.

## Pooled development ranking

MAE weighted by origin counts across years and seeds; pooled RMSE is the square root of origin-weighted squared logged RMSE. Seed SD is descriptive population SD across three pooled seed MAEs. Fold-specific MAEs and seed MAEs in [pooled.csv](pooled.csv); [registry](registry.csv). Rounded source metrics imply rounding-level precision, no naive iid significance. Sort MAE then RMSE; exact joint tie among joint-mlp64, bound-clamp and bound-penalty resolved by sum of logged strategy-fit fit and prediction seconds over all 12 outcomes (including screen): 71.099211, 72.158702 and 72.661672 respectively. These runner-dependent times are noisy, not evidence of a significant speed difference. Cost field populated only for tied recipes; other compute diagnostics remain in archived per-job logs.

| Rank | Method | Recipe | MAE °C | RMSE °C | Seed SD °C |
|---:|---|---|---:|---:|---:|
| 1 | F08 | mae-nhits64 | 1.45243011 | 1.94294115 | 0.00503764 |
| 2 | F09 | huber-nhits64 | 1.45582827 | 1.94537607 | 0.01152534 |
| 3 | F01 | joint-nhits64 | 1.45889978 | 1.94617663 | 0.00878335 |
| 4 | F07 | correction-h16 | 1.50213208 | 1.99681164 | 0.00824073 |
| 5 | F05 | daily-residual-nhits64 | 1.50913773 | 1.99857058 | 0.01797239 |
| 6 | F06 | difference-nhits64 | 1.53901972 | 2.06007591 | 0.01540379 |
| 7 | F05 | daily-residual-mlp64 | 1.54743140 | 2.04848697 | 0.02503854 |
| 8 | F14 | trajectory-curvature | 1.57860087 | 2.09750199 | 0.02461749 |
| 9 | F14 | trajectory-slope | 1.58668946 | 2.10823376 | 0.01440474 |
| 10 | F02 | perlead-h16 | 1.58770531 | 2.12369302 | 0.01920104 |
| 11 | F09 | huber-mlp64 | 1.58883722 | 2.11558114 | 0.01856667 |
| 12 | F11 | lead-early | 1.59087482 | 2.12226729 | 0.01263274 |
| 13 | F01 | joint-mlp64 | 1.59212193 | 2.11688675 | 0.02636705 |
| 14 | F13 | bound-clamp | 1.59212193 | 2.11688675 | 0.02636705 |
| 15 | F13 | bound-penalty | 1.59212193 | 2.11688675 | 0.02636705 |
| 16 | F10 | pinball-multi | 1.59512978 | 2.11781928 | 0.01088301 |
| 17 | F12 | multitask-mean-change | 1.59746084 | 2.12828517 | 0.02999930 |
| 18 | F10 | pinball-median | 1.60293605 | 2.13226018 | 0.02107207 |
| 19 | F08 | mae-mlp64 | 1.60569600 | 2.13697709 | 0.01655254 |
| 20 | F12 | multitask-mean | 1.60591296 | 2.13086120 | 0.02376624 |
| 21 | F02 | perlead-h8 | 1.60651835 | 2.14986689 | 0.02254919 |
| 22 | F11 | lead-late | 1.61633313 | 2.13452984 | 0.02425875 |
| 23 | F07 | correction-affine | 1.63667539 | 2.17258140 | 0.06731848 |
| 24 | F06 | difference-mlp64 | 1.75256434 | 2.33581848 | 0.08625687 |
| 25 | F04 | dirrec-h16 | 2.32806672 | 3.19002455 | 0.22827503 |
| 26 | F04 | dirrec-h8 | 2.34773370 | 3.20768266 | 0.17898219 |
| 27 | F03 | recursive-h16 | 4.95989387 | 6.33051912 | 0.45370130 |
| 28 | F03 | recursive-h32 | 5.37388985 | 6.76819305 | 0.58600038 |

## Decision

Best C010: F08 / mae-nhits64; MAE 1.45243011°C, RMSE 1.94294115°C. Matched-budget F01 joint N-HiTS64 MAE 1.45889978 and joint MLP64 1.59212193. Best-vs-N-HiTS MAE difference -0.00646968°C; best-vs-MLP difference -0.13969182°C. Equal training-origin/epoch budget is not equal compute. C003 full-training MLP64 reference MAE 1.34781461°C; best C010 minus reference 0.10461550°C (7.762%). No C010 candidate beats the full-training reference; the incumbent comparison is descriptive and unequal-budget, not a controlled formulation-only comparison. Historical development periods are not independent holdouts. Retain complete negative results, no extra C010 recipes authorized. Proceed C011 causal same-archive representations; Track B must report intersection and natural coverage separately.

CI: [batch 1](https://github.com/sguzman/aevral/actions/runs/46060337727), [batch 2](https://github.com/sguzman/aevral/actions/runs/46062533543), [batch 3](https://github.com/sguzman/aevral/actions/runs/46065368912), [batch 4](https://github.com/sguzman/aevral/actions/runs/46067195888).
