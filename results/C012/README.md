# C012 optimization results

Frozen scope: [manifest](../../campaigns/012-optimization/manifest.md), [candidate grid](../../campaigns/012-optimization/candidate-grid.csv).
Rung 1: [CI 46081178206](https://github.com/sguzman/aevral/actions/runs/46081178206), code ff7f25565a7feeac83883fdeaefdec5bb27445fb. 48 jobs, 2019 seed42, 2048 rows per candidate. Prepare job passed; evaluation results pending at initialization.

Registry contains one row per audited evaluation, including explicit resource/numerical blocks. A failed infrastructure or audit job is NOT a measured result or a resource block. Reconcile against the frozen matrix and CI job identities before promotion. Retain filtered original logs in rung-specific directories; include GNU time and artifact checksums. Raw predictions stay in 30-day CI artifacts. No duplicate fit on successful audited jobs; recover diagnostics from existing logs if possible.

Registry schema:
- rung,config,fold,seed,track,fit_budget: frozen identity and budget
- origins,mae_c,rmse_c,natural_origins,natural_mae_c,natural_rmse_c: independently audited metrics, blank for blocked outcomes
- fit_s,predict_s,fit_predict_s: cost fields; combined CLI costs are marked in per-job logs, local ridge has no global fitting phase
- parameters: recorded coefficients/leaves/weights, not comparable model complexity across families (local ridge has no global fitted coefficients)
- wall_s,max_rss_kib: GNU time process measurements
- code_sha,ci_run,job_id,status,rescore: provenance; status measured or blocked, rescore ci-rescored or not-applicable
- archive: relative path to retained diagnostics
