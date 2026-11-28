# C015 — Final fixed-data tournament and report

**Status: complete.** All 96 frozen evaluations measured and independently audited. [Final report](REPORT.md) · [Durable numerical archive](../../results/C015/final/) · [Final verification CI](https://github.com/sguzman/aevral/actions/runs/46098369289). P13 achieved 1.29559°C pooled retrospective MAE; no prospective confirmation claimed.

## Scope
Rank all measured configurations and families; top representatives across chronological folds, paired block bootstrap sensitivity, season and horizon diagnostics, performance/compute Pareto front, reproducibility audit and final locked retrospective winner.

## Acceptance
Freeze shortlist and scoring before final fold assessment. Both prior holdouts exposed; only claim retrospective ranking. Optional later prospective confirmation needs genuinely new observations and separate approval.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
