# Roadmap

## C001: Weather / neural fundamentals

Initial milestone (steps 1–7) **complete**. [Final report](campaigns/001-neural-weather/REPORT.md). Step 8 is future research, not part of the completed initial milestone. The 2023–2025 test set is now exposed.

1. Bootstrap ontology, campaign protocol and transparent neural primitives.
2. Acquire and fingerprint annual KSEA observations, 2000–2025. Audit coverage, gaps, report timing and units.
3. Construct leakage-safe hourly examples and chronological partitions; test boundary and missing-data behavior.
4. Evaluate persistence, daily seasonal persistence and training-only climatology.
5. Train a one-hidden-layer MLP with 24 direct outputs and a temperature-only input window. Report MAE/RMSE by lead time and season.
6. Compare input windows and feature groups on matched validation cases: temperature history, calendar, dew point, wind and pressure. Document marginal effects and ablations.
7. Freeze selected model and evaluate the untouched test set once; publish plots, configs and limitations.
8. Compare recurrent, convolutional and attention models, then probabilistic predictions and GPU acceleration.

## Later campaigns

Cross-station transfer; spatial weather inputs; prospective forecast ledger; climate, economics, finance, energy and culture. No commitment to a specific later domain yet.

## Completion criteria for C001 initial milestone

Reproducible data identity and coverage report, passing model/data tests, comparable baseline and neural validation results, locked test evaluation, and a public report. Negative neural skill is a legitimate outcome.
