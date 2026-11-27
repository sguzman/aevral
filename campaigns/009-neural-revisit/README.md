# C009 — Expanded neural architecture tournament

Status: active; authorized as part of [the fixed-data temperature program](../../campaigns/TEMPERATURE-PROGRAM.md). No results yet.

## Scope
MLP width/depth and residual MLP, RNN/GRU/LSTM with longer sequence handling, CNN/TCN, encoder-decoder, N-BEATS/N-HiTS, DLinear/NLinear, TiDE-style dense encoder, patch transformer/PatchTST-style, attention with positional encodings, state-space sequence models, mixture-of-experts.

## Acceptance
Named architectures require accurate implementation or explicit -style label; use original data only; random initialization, no pretrained external weights.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.

[Frozen manifest](manifest.md) · [Screen matrix](screen-matrix.csv): 42 configurations, 17 families; up to 374 additional annual outcomes. No implementation or measurements yet.
