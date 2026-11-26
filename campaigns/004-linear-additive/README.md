# C004 — Linear, regularized and additive models

Status: complete within budget; 54 screening fits, 67 additional measured finalist fits, two explicitly resource-blocked. [Manifest](manifest.md) · [Final report](../../results/C004/REPORT.md) · [Ranking](../../results/C004/pooled.csv). MLP64 remains the best complete four-year model.

## Scope
OLS (QR/SVD or verified stable solver), ridge, lasso, elastic net, Huber, quantile median regression, partial least squares, reduced-rank regression, principal-component regression, polynomial/interactions, GAM/splines, MARS.

## Acceptance
Compare full 80-input direct regression; record rank/conditioning, regularization and loss; test median regression for MAE.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
