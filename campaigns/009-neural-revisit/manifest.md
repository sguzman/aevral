# C009 — frozen expanded neural architecture tournament

Frozen before implementation and outcomes, following C008 closure c03ed04e045b75b9cf7dc0520d38cabbf3b8a596. Track A only: KSEA 72 observed hourly normalized temperatures (oldest→newest) + eight origin calendar/weather features, 24 normalized training targets restored to Celsius. No future observed weather, no extra per-origin history, no pretrained weights. The longer recurrent sequences use smaller patches (up to all 72 hourly steps), not a larger input information set. Use C003 exact annual expanding past-only folds and original full-training seed42 MLP64 comparator; 2019 7455, 2020 7543, 2021 7501, 2022 7639 matched origins. Do not tune on 2023–2026 exposed periods.

## Finite budget and shared training

42 seed42 screens across 17 families, top two per family (34 finalists if all measured). 374 additional annual model/fold/seed outcomes maximum (34 × 11; reuse 2019 seed42). All candidates are stochastic learned-weight models; 2019–2022 with seeds 42,123,2026 for promoted recipes. Select within family on 2019 MAE then RMSE then measured fit+prediction seconds, final exact tie by lexicographic label. No replacement configurations for blocked recipes. Pool within seed with origin-weighted MAE and weighted squared RMSE; across seeds arithmetic mean and sample SD; repeated seeds not independent observations.

Common budget: 8192 deterministic chronological training origins (floor(i*N/n), n=min(N,8192)), all models 12 epochs, one-example Adam update per row, mean squared normalized 24-lead loss, learning rate .001, beta1 .9 beta2 .999 epsilon 1e−8, no weight decay or early stopping. Deterministic Fisher–Yates shuffle per epoch with separate SplitMix64 stream initialized seed XOR 0xD1B54A32D192ED03, high-53-bit uniform, index floor(u*(i+1)); same order across architectures. Xavier uniform sqrt(6/(fan_in+fan_out)) for affine weights, zero bias, learned positional values uniform ±.1, norm gain 1/bias 0, SSM alpha init uniform ±1. Native xorshift64 initializers in pre-existing code may be retained (state=seed.max(1), high53); new modules use that same generator and row-major order; record precise implementation-specific draw order, seeds and parameter counts. Global gradient norm clip 5 for sequence/residual/attention/MoE models (all except N01 plain dense and N11/N12 linear, which use unclipped Adam); track clip counts. No prediction clipping; nonfinite loss/weights/rollout or normalized forecast magnitude >1e6 is explicit blocked. No data-driven adjustment of training budget, epoch count or architecture.

## Frozen architecture recipes

### N01 — MLP
`mlp-w128`, `mlp-w256`, `mlp-128x64`.
Tanh dense layers 80→128→24, 80→256→24, or 80→128→64→24; linear output.

### N02 — Residual MLP
`resmlp-w64-b2`, `resmlp-w128-b2`, `resmlp-w64-b4`.
80→w tanh stem; B blocks h←h+W2 tanh(W1 h+b1)+b2, each w→w→w; linear 24-output head.

### N03 — Elman RNN
`rnn-p6-h16`, `rnn-p3-h16`, `rnn-p1-h16`.
Chronological nonoverlapping patches of p temperatures, h=tanh(Wx+Uh+b), zero state each window, final h plus 8 covariates→24 linear.

### N04 — GRU
`gru-p6-h16`, `gru-p3-h16`, `gru-p1-h16`.
Same patch/readout as N03, reset/update sigmoid gates, reset-after candidate tanh(Wn x+bn+r⊙Un h), h=z⊙h+(1−z)⊙n.

### N05 — LSTM
`lstm-p6-h16`, `lstm-p3-h16`, `lstm-p1-h16`.
Same patch/readout, input/forget/output sigmoid gates and tanh candidate, c=f⊙c+i⊙g, h=o⊙tanh(c), zero initial h/c.

### N06 — CNN
`cnn-p6-c8`, `cnn-p3-c8`.
Three causal convolution layers, kernel 3, dilation 1/1/1, tanh activation, residual addition on equal channel layers after activation as existing Temporal implementation; flattened sequence plus covariates→24.

### N07 — TCN
`tcn-p6-c8`, `tcn-p3-c8`.
As N06, dilations 1/2/4; causal zero left padding, no future taps.

### N08 — Recurrent encoder-decoder
`seq2seq-h16`, `seq2seq-h32`.
Elman tanh encoder over 12 chronological six-temperature patches, final h initializes decoder. Decoder step input [previous scalar normalized temperature, fixed eight covariates, lead/24]; first previous scalar x71, subsequent previous model output. Shared recurrent cell and scalar linear readout, 24 steps, train end-to-end on free-running trajectory (no teacher forcing).

### N09 — N-BEATS-style
`nbeats-w64-b2`, `nbeats-w128-b2`, `nbeats-w64-b4`.
Generic residual backcast/forecast blocks, each receives residual 72-temperature vector plus unchanged covariates, two ReLU hidden layers w, separate linear 72 backcast and 24 forecast heads; subtract backcast and sum forecasts. No interpretable trend/seasonal basis claim.

### N10 — N-HiTS-style
`nhits-w64`, `nhits-w128`.
Three sequential residual blocks, average-pool residual temperature with nonoverlap factors 6,3,1; append eight covariates, two ReLU hidden layers w; linear 72 backcast and forecast knots of length 4,8,24 respectively. Subtract backcasts; interpolate forecasts linearly over endpoints at leads 1 and 24, sum contributions. Simplified pooling/interpolation architecture, not official implementation.

### N11 — DLinear-style
`dlinear-k7`, `dlinear-k25`.
Centered moving average over observed 72-temperature history, odd width k, replicate endpoint padding within history; trend and residual each independent 72→24 affine, plus independent 8→24 linear covariate term. No future padding.

### N12 — NLinear-style
`nlinear-direct`, `nlinear-lead-bias`.
Subtract last observed normalized temperature from all 72 temperatures; concat covariates, linear 80→24, add last back. direct has no head bias; lead-bias has 24 trainable biases. Both variants randomly initialized and trained by Adam.

### N13 — TiDE-style
`tide-w64`, `tide-w128`.
Dense residual encoder 80→w tanh then residual block w→w→w (ReLU inner); bottleneck w→32; decoder 32→w tanh then residual block; linear head 24 plus learned linear skip 72→24. No unavailable future covariate encoder; label style.

### N14 — PatchTST-style
`patchtst-p6-d16`, `patchtst-p3-d16`.
Single temperature channel, nonoverlap p patches, learned patch embedding d=16 and learned absolute positions; one encoder block with two-head scaled dot-product self-attention, output projection, residual then layer norm, FFN 16→32→16 ReLU, residual then layer norm, epsilon 1e−5, trainable affine norms. Flatten tokens, append covariates, 24-output linear head. No official weight claim.

### N15 — Position-aware attention
`attention-p6-d8`, `attention-p3-d8`, `attention-p6-d16`.
Existing compact transformer-style encoder generalized to p,d: learned positions, single-head scaled dot-product attention, residual, tanh feed-forward d→2d→d, flatten all observed tokens and covariates→24; no layer normalization or dropout, distinct from N14.

### N16 — Trainable diagonal state space
`ssm-h16`, `ssm-h32`.
Scalar hourly input, h_i←a_i h_i+(1−a_i)(w_i u+b_i), a_i=sigmoid(alpha_i), 72 steps from zero. Train all decay/drive/head weights by BPTT. Final state plus last normalized temp and covariates→24 affine. Stable diagonal linear state-space sequence model, no S4/Mamba claim.

### N17 — Soft mixture of experts
`moe-e2-w64`, `moe-e4-w64`, `moe-e4-w32`.
E independent 80→w tanh→24 linear experts; 80→E affine softmax gate; weighted sum. Joint end-to-end gradient through every expert and gate, no hard routing/load-balancing claim.

## Execution

Batches: N01–N02 (6), N03–N05 (9), N06–N08 (6), N09–N13 (11), N14–N15 (5), N16–N17 (5). Native Rust, locked Cargo, existing model modules where faithful, explicit implementation-specific -style labels above. New gradient code must pass finite-difference checks on tiny deterministic fixtures for each learned parameter group (relative/absolute tolerance stated), shape/seed repeatability, update and tiny overfit checks, causal input/readout tests and model-specific invariants (recurrence/gate/skip, interpolation, layer norm/softmax, stable decay). Existing generalized modules must retain original behavior/tests for prior recipes; do not modify C003 incumbent or prior results. No undocumented fixed-feature substitutes. If faithful implementation or resource budget fails, record a justified blocked recipe rather than silently simplifying it.

Before each run, analytical workspace bound including weights, Adam state, activation/gradient buffers and cached predictions; <=2GB model workspace. Representative training-only timing probe on 128 selected origins with disposable fresh model/optimizer plus 32 training-input predictions; extrapolate to exact updates and evaluation count, 2x safety +180s setup <=1400s model allowance. Runtime guard checked at each row/update; no partial model scores counted. Job <=30 minutes, max six concurrent fitting jobs across batches, no paid compute. Annual matrices split sequentially into at most 96 jobs per workflow, max six parallel; no overlapping campaign fitting workflows. Archive code/hash/recipe/seed, train count, loss trajectory, clip count, parameter count, preflight, fit/prediction time, peak memory, exact matched score rows and per-lead/month results, compressed predictions retained 30 days. Validate archive hashes, all numerical tests, cohort, complete predictions (24 leads/origin), exact annual origin counts, score/fold/seed consistency. Retry documented implementation/infrastructure defects without altering recipe. Close when finite outcomes measured or explicitly justified blocked; no additional recipes without approval. Then C010.
