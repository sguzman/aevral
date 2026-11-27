# C006 — Trees and boosted ensembles

Status: active; [finite manifest](manifest.md) frozen before fitting. No C006 measurements yet.

## Scope
CART, bagged trees, random forests, Extra Trees, histogram gradient boosting, XGBoost, LightGBM, CatBoost, quantile boosting, multi-output vs horizon-specific trees.

## Acceptance
Backend and dependency pins; no method substitution by name; compare multi-output and per-lead on common origins.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
