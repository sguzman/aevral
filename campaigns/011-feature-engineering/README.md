# C011 — Same-archive representation engineering

Status: active; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
Alternative temperature lags and history lengths, causal rolling mean/range/variance, gradients and acceleration, diurnal harmonics, dew-point depression, pressure tendency, wind direction vectors, meteorological histories from existing columns, seasonal anomalies, causal spectral and interaction features, train-only feature selection.

## Acceptance
Separate expanded-information track from fixed 80-input leaderboard; report matched and natural coverage; no new station, external forecast, or outside data.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.

[Frozen manifest](manifest.md) · [Screen matrix](screen-matrix.csv). 29 recipes across 14 feature families plus one control; 348 maximum measured outcomes. No implementation or measurements yet.
