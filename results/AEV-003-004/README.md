# AEV-003/004 — Calendar features

Code 7bef8b68833d929c0bfb54e9b5471f137b53607e; [successful CI](https://github.com/sguzman/aevral/actions/runs/45865439017). AEV-001 frozen 2000–2022 snapshot verified. Matched 126093 train and 31045 validation origins. No test access.

| Features | Inputs | MAE C | RMSE C | Skill vs daily |
|---|---:|---:|---:|---:|
| Temperature | 72 | 1.59702117 | 2.12970750 | 0.22116366 |
| + UTC hour | 74 | 1.57779346 | 2.10346568 | 0.23054065 |
| + annual phase | 74 | 1.56422061 | 2.08641754 | 0.23715986 |
| + both | 76 | 1.54146757 | 2.06081463 | 0.24825608 |

Daily baseline MAE 2.05052217. Temperature-only reference exactly reproduced AEV-002. Both cycles selected by predeclared aggregate MAE criterion. Relative improvement over matched temperature-only model is about 3.48%. Single seed, overlapping cases and changing input dimensions limit inference; not a significance claim. Full logged records in metrics.txt. Next use the both-cycles configuration as reference on joint eligible meteorological cases.
