# Roadmap

## C001: Weather / neural fundamentals

Initial milestone (steps 1–7) **complete**. [Final report](campaigns/001-neural-weather/REPORT.md). Step 8 architecture comparisons were completed separately in C002; probabilistic predictions and GPU acceleration remain future research. The 2023–2025 test set is now exposed.

1. Bootstrap ontology, campaign protocol and transparent neural primitives.
2. Acquire and fingerprint annual KSEA observations, 2000–2025. Audit coverage, gaps, report timing and units.
3. Construct leakage-safe hourly examples and chronological partitions; test boundary and missing-data behavior.
4. Evaluate persistence, daily seasonal persistence and training-only climatology.
5. Train a one-hidden-layer MLP with 24 direct outputs and a temperature-only input window. Report MAE/RMSE by lead time and season.
6. Compare input windows and feature groups on matched validation cases: temperature history, calendar, dew point, wind and pressure. Document marginal effects and ablations.
7. Freeze selected model and evaluate the untouched test set once; publish plots, configs and limitations.
8. Compare recurrent, convolutional and attention models, then probabilistic predictions and GPU acceleration.

## C002: Architecture comparison

**Complete.** [Final report](campaigns/002-architecture/REPORT.md): dense, recurrent, convolutional and attention models; optimization and seed sensitivity; independent 2026 holdout. Both previous test periods are now exposed.

## C003–C015: Fixed-data temperature research program

**Complete.** [Program and benchmark contract](campaigns/TEMPERATURE-PROGRAM.md) · [Final C015 report](campaigns/015-tournament/REPORT.md) · [Numerical archive](results/C015/final/). All 96 final tournament evaluations measured and independently audited; P13 won with 1.29559°C retrospective pooled MAE. Earlier documented resource blocks remain in the historical records. No additional fixed-data campaign is authorized.

## C016–C018: Deployment bridge

Approved: [reusable Rust interfaces, persistent model artifacts and weather CLI, bounded KLAX deployment and transfer test](campaigns/DEPLOYMENT-PROGRAM.md). C016 pending; completed KSEA research remains frozen. Next-domain proposal: high-dimensional point-in-time macroeconomic forecasting; precise target and manifest pending approval.

## Deferred proposal: Global temperature predictability atlas

[Roadmap and 169-place candidate catalog](campaigns/global-temperature-atlas/ROADMAP.md). Compare absolute forecast error, baseline-relative skill and calibrated uncertainty across climate regimes using frozen reusable models. Planning only; campaign number, station eligibility, execution budget and ordering relative to macroeconomic research await approval. Do not launch until C018 is closed and the user explicitly approves execution.

## Later campaigns

Cross-station transfer; spatial weather inputs; prospective forecast ledger; climate, economics, finance, energy and culture. No commitment to a specific later domain yet.

## Completion criteria for C001 initial milestone

Reproducible data identity and coverage report, passing model/data tests, comparable baseline and neural validation results, locked test evaluation, and a public report. Negative neural skill is a legitimate outcome.
