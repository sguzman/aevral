# C014 — Uncertainty and distributional forecasting

Status: planned; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
Quantile regression, distributional trees, Gaussian/Student-t heads, empirical analog distributions, split conformal with chronological calibration, ensemble distributions.

## Acceptance
Separate objective: pinball loss, CRPS where distribution defined, interval coverage and width; also score point median on MAE. No iid conformal guarantee for dependent time series.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
