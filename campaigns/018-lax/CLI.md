# Weather forecasting CLI

Use the locked Nix environment, including LightGBM 4.6.0 for P13 and coreutils for artifact verification. From the repository root:

```sh
nix develop .#boosting
cargo build --locked --release
export PATH="$PWD/target/release:$PATH"
aevral weather --help
```

For Los Angeles, acquire routine UTC observations in annual immutable slices:

```sh
for y in $(seq 2000 2025); do
  bash scripts/fetch-weather.sh KLAX "$y-01-01" "$((y+1))-01-01" data/raw/lax
done
```

The fetcher validates station identity and source schema, preserves the request URL and retrieval time, and rejects modified cached data. Historical training (example, not a live forecast):

```sh
aevral weather train --station KLAX --model p13 \
  --as-of 2025-01-01T00:00 --calibration-start 2022 --calibration-end 2024 \
  --artifact models/KLAX-2025-p13.aev
aevral weather forecast --station KLAX --model p13 --hours 24 \
  --as-of 2025-01-02T00:00 --artifact models/KLAX-2025-p13.aev
```

For a current forecast, extend the training archive through the desired completed UTC hour with the fetcher's optional explicit cutoff. The exclusive end date is the following UTC date. For example, in 2026:

```sh
issue=$(date -u +%FT%H:00)
bash scripts/fetch-weather.sh KLAX 2026-01-01 \
  "$(date -u -d "${issue:0:10} +1 day" +%F)" data/raw/lax "$issue"
aevral weather train --station KLAX --model p13 --as-of "$issue"
```

Do not mix overlapping snapshots in the same directory when refreshing current observations. For inference, a separate fresh directory with the last four complete dates plus the current partial date suffices:

```sh
issue=$(date -u +%FT%H:00)
bash scripts/fetch-weather.sh KLAX "$(date -u -d "${issue:0:10} -4 days" +%F)" \
  "$(date -u -d "${issue:0:10} +1 day" +%F)" fresh "$issue"
aevral weather forecast --station KLAX --hours 24 --as-of "$issue" --data fresh
```

Use a new directory for each issue or remove/archive the old observation snapshot first. Forecasts require 72 consecutive hourly temperature bins and complete origin weather covariates; unavailable data is an error rather than a backdated or imputed forecast. The CLI never overwrites an artifact or output file. To use the original MLP, specify `--model mlp` for both training and forecasting; interval columns will be empty. All values are Celsius, dates UTC, and lead hours 1–24. The artifact includes the training cutoff, source hashes, transforms, versioned recipe, calibration years, backend and inference parameters. It is not an optimizer checkpoint. For pinned P13 inference remain in the boosting shell. KSEA uses its existing historical conflict policy; KLAX uses strict ingestion.

The default P13 calibration is the last three complete calendar years, not earlier than 2020. Explicit years are recommended for reproducing an assessment. This is an observational forecasting tool, not a substitute for official weather warnings. See [the frozen assessment protocol](manifest.md).
