# AEV-015 — Frozen independent 2026 evaluation

[Passing scoring CI](https://github.com/sguzman/aevral/actions/runs/45880217793), code 2e816e47d038a9e37cce68f056fd4cabb2297e32. [Frozen recipe](../../campaigns/002-architecture/FROZEN.md). [Snapshot](snapshot.sha256), [provenance](snapshot.txt), [pre-score audit](audit.txt), [all 256 records](metrics.txt).

Targets wholly within January–October 2026 UTC; 6,521 matched origins of 7,273 possible (89.66%), 156,504 target predictions per model. 30 implementation tests passed. Original training and validation counts and normalization reproduced exactly; no new timestamp conflicts. Both validation MAEs reproduced before scoring.

| Forecast | MAE (°C) | RMSE (°C) | MAE reduction vs daily |
|---|---:|---:|---:|
| Winner MLP64 | 1.352640 | 1.796293 | 37.67% |
| Original MLP32 | 1.377902 | 1.825811 | 36.51% |
| Daily persistence | 2.170267 | 2.843450 | 0.00% |
| Climatology | 2.435842 | 3.168644 | -12.24% |
| Current persistence | 3.255039 | 4.331375 | -49.98% |

Winner MAE reduction versus original: 1.833304%. RMSE falls from 1.82581070 to 1.79629347°C. Monthly comparison (month of first target):

| Month | Origins | Original MAE | Winner MAE |
|---|---:|---:|---:|
| 1 | 639 | 1.227585 | 1.242480 |
| 2 | 663 | 1.259326 | 1.226542 |
| 3 | 721 | 1.414839 | 1.410948 |
| 4 | 601 | 1.405892 | 1.358696 |
| 5 | 599 | 1.683834 | 1.656859 |
| 6 | 696 | 1.512521 | 1.467636 |
| 7 | 609 | 1.439932 | 1.442548 |
| 8 | 704 | 1.286259 | 1.221518 |
| 9 | 693 | 1.215591 | 1.207061 |
| 10 | 596 | 1.366972 | 1.326627 |

Winner lower in 8 of 10 months. This is historical evaluation at one station, with overlapping forecast windows and one preselected seed for each neural model. No significance claim, no prospective operational forecast, no post-test tuning. The 2026 holdout is now exposed; reserve a new test for subsequent selection. Raw data remain in CI cache, not Git; hashes verify identity, not indefinite cache availability.
