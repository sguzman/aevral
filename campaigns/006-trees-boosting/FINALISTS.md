# C006 frozen annual finalists
2019 screen 28/28 measured. Promote top two per family by MAE then RMSE (shared and per-lead CART distinct subfamilies). Reuse 2019 deterministic results and stochastic seed42 results. Every finalist refits on the same fixed 24576 chronological training-origin sample per fold. Annual folds 2019–2022, all evaluation origins. No later-fold tuning.
| Family | Finalists | Seeds |
|---|---|---|
| T01 shared | cart-shared-d8-l32, cart-shared-d8-l128 | deterministic (seed42 where applicable) |
| T01 per-lead | cart-lead-d8-l32, cart-lead-d4-l32 | deterministic (seed42 where applicable) |
| T03 | rf-32-27, rf-16-27 | 42, 123, 2026 |
| T04 | extra-32-27, extra-16-27 | 42, 123, 2026 |
| T02 | bag-32, bag-16 | 42, 123, 2026 |
| T05 | hist-64-d5, hist-32-d5 | deterministic (seed42 where applicable) |
| T09 | median-boost-64, median-boost-32 | deterministic (seed42 where applicable) |
| T07 | lightgbm-128, lightgbm-64 | deterministic (seed42 where applicable) |
| T06 | xgboost-128, xgboost-64 | deterministic (seed42 where applicable) |
| T08 | catboost-128, catboost-64 | 42, 123, 2026 |
20 point finalists, 12 deterministic and 8 stochastic. Remaining 124 evaluations: 36 deterministic (3 additional folds ×12) + 88 stochastic (12 fold-seed pairs per config minus reused 2019 seed42). Use all four folds for each seed; report pooled MAE/RMSE separately by seed, mean and SD across seeds, no pooling seeds as extra independent observations. C003 seed42 incumbent reused. Group by fold and module/seed, max six parallel, <=30min/job; avoid launching multiple workflows concurrently beyond six fit jobs. Preserve all records, per-lead and month metrics, seed-specific artifacts, timings and resource counts. Explicitly report different training budgets. No new data.
