# C012 — Bounded optimization and annual finalist comparison

**Result.** LightGBM is the best complete-cohort finalist by pooled matched MAE and RMSE in this development experiment. N-HiTS has lower available-case errors but three resource-blocked fits, so it is not eligible for the complete-cohort ranking. This is not an independent test result.

## Protocol and accountability

The [frozen manifest](manifest.md) specified eight families, six recipes each, successive halving on 2019 at 2,048 / 8,192 / 32,768 training origins, then annual comparison on 2019–2022 at the winning recipe and largest budget. Every scheduled outcome is in the [registry](../../results/C012/registry.csv). Rungs 1/2/3 yielded 48/24/8 measured outcomes; 24 recipes were pruned after rung 1 and 16 after rung 2. Final annual additions yielded 45 measured and three blocked outcomes. Total: 128 scheduled, 125 measured, three blocked. Eight rung-3 scores are reused, not refit; annual comparison has 56 planned and 53 measured entries. No substitution or additional scored fits. The [candidate grid](candidate-grid.csv), promotion matrices, [rung-3 ranking](../../results/C012/rung-3-rankings.csv) and [final matrix](rung-final.csv) retain the selection record.

All finalists use the same matched origins: 7,455 / 7,543 / 7,501 / 7,639 by year (30,138 total). Within each year, average MAE and squared RMSE equally over measured seeds; pool across years by origin count, then take the square root for RMSE. Deterministic models have one seed per year. These are averages of individual model errors, not ensemble predictions. The A-derived shortened-history model uses the same information type with fewer temperature lags. The 2019 selection fold is included in the comparison; 2023–2026 are not used here.

## Annual results

| Complete rank | Model | Config | Track | Measured/planned | MAE °C | RMSE °C | Median fit+predict s |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | LightGBM | O04-04 | A-fixed | 4/4 | 1.366640 | 1.843701 | 39.716 |
| 2 | Local ridge | O05-02 | A-fixed | 4/4 | 1.375004 | 1.846745 | 20.207 |
| 3 | MLP80 | O01-06 | A-fixed | 12/12 | 1.378863 | 1.845655 | 22.183 |
| 4 | Suffix24 MLP | O02-06 | A-derived | 12/12 | 1.428968 | 1.906709 | 10.329 |
| 5 | MARS | O06-06 | A-fixed | 4/4 | 1.483253 | 1.981328 | 0.587 |
| 6 | Haar ridge | O08-06 | A-fixed | 4/4 | 1.557362 | 2.061522 | 0.211 |
| 7 | ARX | O07-06 | A-fixed | 4/4 | 1.590479 | 2.105016 | 0.612 |
| — | MAE N-HiTS | O03-04 | A-fixed | 9/12 | 1.350132 | 1.825832 | 438.586 |

N-HiTS has 2/3, 3/3, 1/3 and 3/3 successful seeds by year. Its three blocks (2019/2026, 2021/42, 2021/123) hit the predeclared runtime guard at 1,400.92, 1,401.10 and 1,400.82 seconds elapsed, respectively. Missing results are not assumed random, imputed, or ranked as complete; its successful median cost excludes blocked time. No reruns. The apparent advantage over LightGBM is descriptive only. Median costs describe these CI runs and include model-specific fit/predict conventions; runner performance varies. All resource records are retained in the registry; measured peak process RSS is about 320 MiB or less.

LightGBM improves pooled MAE over the optimized MLP by about 0.89%, at about 1.79 times median fit+predict cost. Local ridge is within 0.62% of LightGBM's MAE with about half the median cost, but its prediction cost dominates. MARS is a fast lower-accuracy alternative; Haar ridge is the fastest measured median. The historical C003 full-training 2019 MLP reference (1.32617622°C MAE) is separate from the 32,768-origin C012 comparison: none of these budgeted 2019 finalists establishes improvement over it.

## Variability and coverage

| Model | 2019 MAE | 2020 MAE | 2021 MAE | 2022 MAE |
|---|---:|---:|---:|---:|
| MLP80 | 1.36530 | 1.33739 | 1.41594 | 1.39664 |
| Suffix24 MLP | 1.39781 | 1.37112 | 1.48869 | 1.45785 |
| N-HiTS (available-case) | 1.34285 | 1.31502 | 1.38827 | 1.35446 |
| LightGBM | 1.34042 | 1.33239 | 1.40497 | 1.38841 |
| Local ridge | 1.35130 | 1.33089 | 1.42571 | 1.39190 |
| MARS | 1.43487 | 1.45931 | 1.54577 | 1.49273 |
| Haar ridge | 1.47352 | 1.53459 | 1.60416 | 1.61572 |
| ARX | 1.51602 | 1.56499 | 1.63190 | 1.64765 |

The complete stochastic MLP seed ranges vary by year; largest within-year MAE span is about 0.0360°C (2022) for MLP80 and 0.0482°C (2019) for Suffix24. The single successful 2021 N-HiTS seed cannot characterize seed variability. See [fold table](../../results/C012/annual-folds.csv) for all ranges and seed counts; no uncertainty intervals or significance claims based on these overlapping forecast targets.

Suffix24 natural coverage is 31,956 origins (1,818 additional, 6.03% more than matched), with MAE 1.42770223 and RMSE 1.90378760°C, versus 1.42896796 and 1.90670899 on matched origins. Natural results are not cross-model matched comparisons.

## Lead and seasonal diagnostics

| Model | 1h MAE | 12h MAE | 24h MAE | June MAE |
|---|---:|---:|---:|---:|
| LightGBM | 0.53767 | 1.44482 | 1.71577 | 1.50058 |
| Local ridge | 0.57300 | 1.45988 | 1.72299 | 1.47626 |
| MLP80 | 0.64513 | 1.45356 | 1.71878 | 1.51977 |
| Suffix24 MLP | 0.72133 | 1.49545 | 1.76119 | 1.59192 |
| N-HiTS (available-case) | 0.63356 | 1.41472 | 1.71519 | 1.45822 |
| MARS | 0.55870 | 1.56184 | 1.79420 | 1.63209 |
| Haar ridge | 0.58172 | 1.69526 | 1.84753 | 1.75440 |
| ARX | 0.56269 | 1.72319 | 1.94600 | 1.76710 |

June has highest pooled monthly MAE in each matched cohort (2,405 June origins). No causal interpretation follows from this observation. All 24 leads and 12 months are in [annual diagnostics](../../results/C012/annual-diagnostics.csv), including separate natural Suffix24 diagnostics. The diagnostic aggregator checks that each archive's lead and month errors reproduce its score within 1e-7.

## Reproduction and next campaign

From repository root in the locked Rust development shell:

```sh
nix develop --command rustc --edition=2021 -O campaigns/012-optimization/aggregate-final.rs -o /tmp/c012-aggregate
/tmp/c012-aggregate results/C012/registry.csv campaigns/012-optimization/rung-final.csv /tmp/c012-annual
nix develop --command rustc --edition=2021 -O campaigns/012-optimization/aggregate-diagnostics.rs -o /tmp/c012-diagnostics
/tmp/c012-diagnostics results/C012 /tmp/c012-annual
cmp /tmp/c012-annual/annual-summary.csv results/C012/annual-summary.csv
cmp /tmp/c012-annual/annual-folds.csv results/C012/annual-folds.csv
cmp /tmp/c012-annual/annual-diagnostics.csv results/C012/annual-diagnostics.csv
```

The 2,340 fold/seed diagnostic rows are regenerated from archived evidence rather than duplicating them in Git. [Summary CI](https://github.com/sguzman/aevral/actions/runs/46088810645) and [diagnostic CI](https://github.com/sguzman/aevral/actions/runs/46089035095) reproduced the outputs. [Final scoring run](https://github.com/sguzman/aevral/actions/runs/46085630766). Per-fit logs retain data/fit/cohort provenance and prediction checksums; large prediction artifacts are subject to CI retention, not permanently stored in Git.

C013 will investigate hybrids and ensembles under its own frozen protocol. The observed differences do not justify claims of general superiority or independent validation. C012's resource cap and incomplete N-HiTS coverage are material limitations.
