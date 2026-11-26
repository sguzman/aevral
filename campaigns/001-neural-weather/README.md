# C001 — Neural weather forecasting

Question: How much of the next 24 hours of local air temperature can be inferred from past local observations alone?

Station: KSEA (IEM SEA, WA_ASOS). Target: near-surface air temperature in Celsius. Time axis: UTC hourly bins. At each integer-hour origin t, use the most recent routine observation in each completed interval (t-1h,t], retaining its actual timestamp. Choose the latest report in a bin before checking field validity; missing values remain missing. Output 24 direct targets for bins ending at t+1h through t+24h. The target is the last routine observation in each respective interval, not instantaneous temperature at the nominal hour. Archive publication latency is unknown; retrospective event-time approximation only.

Initial window: 24 completed hourly bins ending at t. Compare 6/12/24/48/72 hours later. A calendar-day forecast is a separate later task (local timezone and DST explicitly handled).

## Experimental design

- Candidate raw period: 2000-01-01 through 2025-12-31 UTC. Confirm usable coverage before running.
- Train target times before 2019-01-01; validation 2019–2022; locked test 2023–2025. Exclude any example with targets spanning a split boundary. Historical context preceding a split is allowed.
- Baselines: last known temperature, same-hour previous day (for each lead), and training-only hourly seasonal climatology.
- AEV-001: temperature-only MLP, 24 temperature inputs, 32 tanh hidden units, 24 linear outputs, MSE loss; train-only standardization; seed 42. CPU first.
- Missingness: no forward or backward filling in the initial strict complete-window experiment; compare all methods on identical origins. Report excluded origins and coverage. If inadequate, predeclare a new causal missingness policy before scoring.
- Metrics: MAE and RMSE in °C by lead, aggregate MAE, skill relative to daily persistence, sample count and seasonal slices. Do not claim superiority to operational numerical weather forecasts.
- Feature campaign: add calendar cycles, dew point, wind speed/direction (vector representation), altimeter pressure and other documented observations individually and jointly. Keep an equal-origin comparison; report coverage and cost. Tune only on validation.
- Test lock: do not inspect test scores during feature or architecture selection. Record code commit, raw SHA-256, configuration, seeds and results.

The experiment is an observational benchmark, not an assertion of physical causality or operational forecast readiness.
