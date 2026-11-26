# Aevral

An open research laboratory for forecasting: identifying useful observations, learning temporal relationships, and measuring predictive skill across domains.

The first campaign studies neural networks for hourly air-temperature forecasting at Seattle–Tacoma International Airport (KSEA). Aevral is not restricted to neural networks or weather.

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

Rust is the primary language. The educational neural-network core is dependency-free; Burn and GPU backends are later comparisons. A Nix development shell is provided (lockfile pending).

```sh
nix develop
cargo test
cargo run --release -- smoke
```

The smoke command learns synthetic data only; it is not a weather result. Real observations have not yet been acquired or evaluated.
