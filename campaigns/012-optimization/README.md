# C012 — Budgeted cross-family hyperparameter search

Status: **complete** (125 measured outcomes, three documented runtime blocks, all 128 scheduled outcomes resolved). [Final report](REPORT.md) · [Frozen manifest](manifest.md) · [Candidate grid](candidate-grid.csv) · [Annual summary](../../results/C012/annual-summary.csv) · [Reproduction CI](https://github.com/sguzman/aevral/actions/runs/46089192627).

## Scope
Learning rates, schedules, optimizers, batch sizes, weight decay, dropout, early stopping, tree depths and leaves, regularization, lag/order, kernel scales, neighbor distances; deterministic random search or successive halving, then multi-seed finalists.

## Acceptance
Predeclare finite grids and budgets before launch; search only development folds; report all trials and pruning; compare budget-normalized and best attainable scores.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
