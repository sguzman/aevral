# Aevral

An open research laboratory for forecasting: identifying useful observations, learning temporal relationships, and measuring predictive skill across domains.

The first campaign studies neural networks for hourly air-temperature forecasting at Seattle–Tacoma International Airport (KSEA). Aevral is not restricted to neural networks or weather.

## First research milestone completed

**1.363°C MAE on the independent 2023–2025 KSEA test**, 33.34% lower than daily persistence, across 23,799 matched origins. See the [campaign report](campaigns/001-neural-weather/REPORT.md), [all experiments](results/README.md), and [independent evaluation](results/AEV-009/README.md).

![Independent forecast errors by horizon](results/AEV-009/lead-mae.svg)

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
