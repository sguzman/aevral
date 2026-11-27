# C008 — Decomposition, dynamical systems and alternative learners: closed

36 screening outcomes plus 86 additional annual outcomes, all 122 measured, zero blocked. 18 finalists in nine families. Annual CI [46033359536](https://github.com/sguzman/aevral/actions/runs/46033359536) all 86 passed; 145 tests. Each seed pool uses 30,138 matched annual origins (723,312 target hours), 2019–2022; repeated seeds are not independent observations. MAE origin-weighted; RMSE square root of origin-weighted annual squared RMSE. Stochastic summaries are arithmetic means of three seed pools with sample SD. Calculated from rounded archived annual scores, not independently rescored raw predictions. Training budgets and information sets follow frozen manifest; C003 incumbent uses full training. C003 incumbent MAE 1.34781461, RMSE 1.80956416. Best C008 haar-m32-l001 MAE 1.55757204, 15.56% higher than incumbent. No incumbent change.

| Rank | Model | Family | Seeds | MAE °C | RMSE °C | MAE seed SD | RMSE seed SD |
|---:|---|---|---:|---:|---:|---:|---:|
| — | C003 MLP64 | incumbent | 1 | 1.34781461 | 1.80956416 | — | — |
| 1 | haar-m32-l001 | D02 | 1 | 1.55757204 | 2.06292167 | — | — |
| 2 | fourier-k6-l001 | D01 | 1 | 1.55799940 | 2.06268474 | — | — |
| 3 | elm-m128-l001 | D07 | 3 | 1.55845344 | 2.05923388 | 0.00724821 | 0.01051122 |
| 4 | haar-m32-l01 | D02 | 1 | 1.56128574 | 2.06575120 | — | — |
| 5 | fourier-k6-l01 | D01 | 1 | 1.56709113 | 2.06983286 | — | — |
| 6 | elm-m128-l01 | D07 | 3 | 1.58327656 | 2.09162432 | 0.00797252 | 0.01186515 |
| 7 | symbolic-depth1-t8 | D09 | 1 | 1.64100702 | 2.15051428 | — | — |
| 8 | symbolic-depth2-t8 | D09 | 1 | 1.64100702 | 2.15051428 | — | — |
| 9 | esn-m128-leak03 | D06 | 3 | 1.67074818 | 2.20371172 | 0.00693135 | 0.00913342 |
| 10 | esn-m64-leak03 | D06 | 3 | 1.68566950 | 2.22367162 | 0.00360296 | 0.00629741 |
| 11 | dmd-l24-r8 | D04 | 1 | 1.74114874 | 2.31275314 | — | — |
| 12 | ssa-l24-r8 | D03 | 1 | 1.83063346 | 2.40785704 | — | — |
| 13 | dmd-l24-r4 | D04 | 1 | 1.90180119 | 2.47828008 | — | — |
| 14 | ssa-l36-r8 | D03 | 1 | 1.91412300 | 2.53623906 | — | — |
| 15 | markov-b16-hour | D08 | 1 | 2.05014269 | 2.77904881 | — | — |
| 16 | edmd-l12-l001 | D05 | 1 | 2.14023776 | 2.82799243 | — | — |
| 17 | edmd-l12-l01 | D05 | 1 | 2.24293779 | 2.97087743 | — | — |
| 18 | markov-b8-hour | D08 | 1 | 2.32707168 | 3.17463340 | — | — |

Fourier and Haar offer low-dimensional causal projections; the recurrent and lifted dynamics are finite approximations, not physical system identification. ESN and ELM are neural/hybrid learners. Symbolic depth1 and depth2 eight-term recipes tie across all four annual aggregate scores; depth1 is cheaper in screening. Retain both as frozen finalists, not independent evidence. Hour-conditioned observable Markov transition models were promoted over pooled ones.

[Registry](registry.csv) · [Seed pools](pooled-seeds.csv) · [Rankings](pooled-rankings.csv) · [Closure audit](closure-audit.md) · [Frozen matrix](../../campaigns/008-decomposition-reservoir/finalist-matrix.csv).

Next: C009 expanded neural architecture tournament; freeze finite manifest before implementation.
