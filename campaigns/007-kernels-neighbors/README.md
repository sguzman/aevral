# C007 — Kernel, local and instance-based methods

Status: planned; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
KNN, weighted analogs, nearest seasonal trajectories, local linear regression, RBF/linear kernel ridge, epsilon-SVR, Gaussian processes, sparse/inducing point approximations, random Fourier/Nystrom features.

## Acceptance
Avoid full O(n²) memory or O(n³) fitting at 121849 origins; scalable variants use documented train-only approximations; record exact vs approximate.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
