# C006 — Trees and boosted ensembles

Status: planned; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
CART, bagged trees, random forests, Extra Trees, histogram gradient boosting, XGBoost, LightGBM, CatBoost, quantile boosting, multi-output vs horizon-specific trees.

## Acceptance
Backend and dependency pins; no method substitution by name; compare multi-output and per-lead on common origins.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
