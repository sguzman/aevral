# C002 — Neural architecture comparison

Goal: improve the C001 80-input direct 24-hour KSEA temperature forecast by comparing architectures, not changing the observations initially.

**Completed:** [final report](REPORT.md), [independent evaluation](../../results/AEV-015/README.md). Protocol below is retained as the original plan; the holdout has now been evaluated and is exposed.

## Protocol
- Preserve C001 unchanged as historical benchmark; 2023–2025 is exposed and never described as independent again.
- Use the same frozen 2000–2022 data, 121849 training and 30207 validation origins, same 80 inputs, train-only scaling and 24 output labels.
- First screen candidates with seed 42, 10 epochs and SGD 0.005, then investigate optimizer and training-budget sensitivity. Report parameter counts, wall time, aggregate and per-lead MAE/RMSE. Fixed training budget is a controlled comparison, not proof of optimal architecture.
- Evaluate promising alternatives across predeclared multiple seeds, publish unfavorable outcomes, and select by validation MAE and robustness. Keep C001 reference and daily persistence.
- New holdout designated now: KSEA target hours from 2026-01-01 00:00 UTC to 2026-11-01 00:00 UTC (exclusive), not yet accessed by this project. Acquire only after final model and evaluation protocol freeze. Historical holdout, not prospective forecast. Earlier history may provide causal context; no fitting from the new holdout. If archive availability or data-quality issues block the test, record them before scoring, not change the date range based on results.
- Keep CPU Rust implementations and reproducible CI initially; use Nix/CUDA or optimized backend when scale requires it. Do not compare unoptimized execution time as inherent architecture cost.

## Sequence
1. AEV-010: linear SGD comparator and dense width/depth screening.
2. AEV-011: simple recurrent network, GRU and LSTM with explicit sequence handling and gradient checks.
3. AEV-012: temporal convolution and dilated TCN.
4. AEV-013: attention/transformer, same historical observations and origin covariates.
5. AEV-014: fair multi-seed and training-budget comparisons of leading candidates, including original MLP; freeze winner.
6. AEV-015: independent 2026 holdout evaluation and public report.

Acceptance: actual matched validation results for architecture families, tested implementations, transparent comparisons of optimization and capacity, frozen new holdout result, report and durable checkpoint. No requirement that complex models win. Future feature expansion or deployment separate.
