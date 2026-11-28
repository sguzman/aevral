# Forecasting deployment bridge (C016–C018)

Approved 2026-11-27: reusable Rust tool and practical weather train/forecast commands, with LAX as the first cross-station deployment. C001–C015 remain closed and their frozen historical measurements must not be modified.

## C016 — Extract reusable forecasting interfaces
- Separate station identity, causal feature construction, training-only preprocessing, fit, predict, and scoring from the KSEA experiment commands. Preserve historical entry points and test parity with C015.
- Introduce station-validated ingestion for IEM-format observations, no silently relabeled station IDs. Retain UTC, missing observations, duplicate/conflict handling and strict causal availability.
- Expose stable native Rust training and inference interfaces for incumbent I (80→64→24 SGD MLP) and the four P13 members. Reuse numerical recipes rather than approximating them.
- Tests: synthetic station identity and causal-origin checks; independent prediction parity on existing frozen examples. No new KSEA tuning.

## C017 — Durable models and CLI
- Implement versioned model artifacts with recipe, station, fit cutoff, feature schema, transforms, training data fingerprints and backend/version metadata. Validate malformed/incompatible/nonfinite artifacts; roundtrip exact inference parity.
- CLI contract approved: `aevral weather train --station KLAX --model p13` and `aevral weather forecast --station KLAX --hours 24`. Also support incumbent MLP; specify data/model paths and as-of cutoff with clear defaults/help. Forecasts issue at a completed causal origin, use only available observations, and reject unavailable inputs; P13 exports median and uncertainty. Preserve current research commands.
- P13 requires four independently trained base learners (I, LightGBM, local ridge, Adam MLP) plus prior-only calibration; no claim that saving only tree parameters saves a usable ensemble. Pin LightGBM backend and model loading in Nix; verify artifact roundtrip for each component.
- Rust-first; no paid service, new dependency or network assumption without justification. Document local cargo/Nix usage, data access, licensing and provenance. No fake live results.

## C018 — LAX deployment and bounded transfer test
- Discover/verify source identity (IEM LAX / ICAO KLAX) and observation schema; obtain and fingerprint a separate historical data snapshot. Document availability and coverage before training. Do not silently mix with KSEA.
- Compare frozen KSEA-trained incumbent transfer, LAX-trained incumbent and LAX-trained P13 with prior-only LAX calibration, matched chronological assessment, persistence reference, MAE/RMSE and uncertainty. Fit the Seattle artifact with an explicit pre-assessment cutoff. Freeze dates and budgets before assessment; no new exhaustive weather tournament.
- Produce a real as-of next-24-hour forecast on KLAX with provenance and timestamp if fresh complete observations are available; otherwise report data limitation, not fabricated temperatures. Provide commands and CSV forecast outputs, archive verification and results.
- Stop once the CLI, artifacts, and bounded cross-station test are verified. Separate approval needed for new research domain and detailed candidate budget.

## Next domain proposal (not yet frozen)
High-dimensional macroeconomic/financial forecasting, starting with a single economic target such as monthly US industrial production growth, with expanding predictor groups and point-in-time data. FRED-MD historical vintages: https://www.stlouisfed.org/research/economists/mccracken/fred-databases ; ALFRED: https://alfred.stlouisfed.org/. Explore target-only baselines, macro factor models and regularized multi-input models, staged feature-group ablations, data revision and publication-lag controls, then ensembles and probabilistic prediction. Target, horizons, vintage availability, finite scope and held-out dates must be frozen before assessment. No finance/market trading or further new data ingestion authorized by this proposal alone.
