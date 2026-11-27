# C010 — Forecast formulation and objective

Status: active; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
Direct joint 24-output vs per-lead vs recursive vs DirRec, daily residuals, differences, learned corrections, MAE/Huber/MSE/quantile loss, lead-weighted losses, multi-task regularization, physical bounds and trajectory penalties.

## Acceptance
Use top representative families; no future realized covariates for recursive prediction; score original Celsius target and same 24 leads.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.

[Frozen manifest](manifest.md) · [Screen matrix](screen-matrix.csv). 28 recipes across 14 method groups; no implementation or measurements yet.
