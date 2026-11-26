# C012 — Budgeted cross-family hyperparameter search

Status: planned; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
Learning rates, schedules, optimizers, batch sizes, weight decay, dropout, early stopping, tree depths and leaves, regularization, lag/order, kernel scales, neighbor distances; deterministic random search or successive halving, then multi-seed finalists.

## Acceptance
Predeclare finite grids and budgets before launch; search only development folds; report all trials and pruning; compare budget-normalized and best attainable scores.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
