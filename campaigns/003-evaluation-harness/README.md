# C003 — Common benchmark and reproducibility

Status: planned; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
Export matched origins; verify incumbent and naive baselines; chronological fold builder; leakage and missingness tests; manifest, deterministic scoring, coverage and CI resource accounting; ranking ledger.

## Acceptance
No search until exact C002 validation reproduction; fold and budget manifest fixed.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
