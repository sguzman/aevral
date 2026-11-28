# Aevral

An open research laboratory for forecasting: identifying useful observations, learning temporal relationships, and measuring predictive skill across domains.

The first campaign studies neural networks for hourly air-temperature forecasting at Seattle–Tacoma International Airport (KSEA). Aevral is not restricted to neural networks or weather.

## First research milestone completed

**1.363°C MAE on the independent 2023–2025 KSEA test**, 33.34% lower than daily persistence, across 23,799 matched origins. See the [campaign report](campaigns/001-neural-weather/REPORT.md), [all experiments](results/README.md), and [independent evaluation](results/AEV-009/README.md).

![Independent forecast errors by horizon](results/AEV-009/lead-mae.svg)

## Second research milestone completed

**1.353°C MAE on the independent January–October 2026 holdout**, 1.83% lower than the original MLP and 37.67% lower than daily persistence, over 6,521 matched origins. Eleven initial architecture configurations and a 36-run multi-seed optimization comparison. [Campaign report](campaigns/002-architecture/REPORT.md) · [Independent results](results/AEV-015/README.md).

![2026 forecast errors by horizon](results/AEV-015/lead-mae.svg)

## Active research program

[C003–C015: comprehensive fixed-data temperature forecasting](campaigns/TEMPERATURE-PROGRAM.md). Same KSEA archive, broad model families, controlled rolling-origin development folds, bounded optimization and final retrospective ranking. C003–C014 complete; C015 final retrospective tournament active. [C013 report](campaigns/013-ensembles/REPORT.md): four-model ensemble 1.35048°C pooled development MAE. [C014 report](campaigns/014-probabilistic/REPORT.md): distributional tree 0.97157°C CRPS and 89.86% empirical coverage at nominal 90%. These are development results, not independent holdout confirmation.

## Research principles

- Define the forecast origin, target, horizon and information available at issue time.
- Compare every model to transparent baselines on the same cases.
- Freeze chronological train/validation/test splits; never tune on the test set.
- Publish failed experiments alongside successful ones.
- Cache raw data locally, record provenance and checksums, and keep large artifacts out of Git.
- Separate real historical backtests, prospective forecasts, and synthetic implementation checks.

## Start here

- [Ontology](ONTOLOGY.md)
- [Research roadmap](ROADMAP.md)
- [Campaign 001](campaigns/001-neural-weather/README.md)
- [Experiment 001](experiments/AEV-001.md)
- [Dataset manifest](domains/weather/ksea.md)
- [Continuation state](STATE.md)

## Development

Rust is the primary language. The educational neural-network core is dependency-free; Burn and GPU backends are later comparisons. A verified, locked Nix development shell is provided.

```sh
nix develop
cargo test
cargo run --release -- smoke
```

The smoke command learns synthetic data only; it is not a weather result. Real-data experiments and reproducibility instructions are in the campaign report.
