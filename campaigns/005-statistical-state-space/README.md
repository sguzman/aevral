# C005 — Classical statistical and state-space forecasting

Status: active; [finite manifest](manifest.md) frozen before fitting. No C005 results yet.

## Scope
AR/ARX, VAR/VARX where causal histories available, ARIMA/ARIMAX, seasonal ARIMA, ETS/Holt-Winters, harmonic regression with AR errors, structural/unobserved components, Kalman dynamic linear model, switching/regime models, GARCH-style conditional variance (uncertainty only).

## Acceptance
Separate fixed-window and archive-trained state variants; no future covariates; test filtering vs smoothing causality, no test target assimilation before issue time.

Common requirements: versioned configuration manifest before runs; actual matched-origin predictions and aggregate/lead/fold metrics or explicit blocked/skipped status with reason; tested implementation, frozen data hashes, recorded code/backend/seed/budget, CI links and durable results. No undocumented substitutions. No new external data. Follow program gates and checkpoint STATE.md at every verified milestone. Stop at this campaign's predeclared finite budget, even if no improvement.
