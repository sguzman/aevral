# C013 — Cross-family hybrids and ensembles

Status: planned; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
Mean/median, nonnegative convex weights, horizon-specific weights, ridge stacking, residual correction, regime-gated mixtures, diversity-aware ensembles, calibrated blending.

## Acceptance
Stack only out-of-fold predictions; fit weights before final assessment; compare gains against best single model and added cost.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
