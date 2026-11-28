# C014 forecast export format

Each annual output has 24 gzip files P01–P24, one for each frozen recipe, plus matching parameter CSVs, metrics.csv and costs.csv. Gzip is streamed; no temporary uncompressed prediction file is required. Artifact retention is finite; archive aggregate metrics, fitted parameters, hashes and reproduction provenance in Git.

## Binary v1

All numeric fields little-endian. Header: 8 ASCII bytes C014BIN1, u16 assessment year, u8 recipe ID. Each subsequent fixed-size record corresponds to one source row in original origin-major, lead-minor order.

| Field | Type/count |
|---|---|
| Origin UTC minutes since 2000 | i64 |
| Lead (1–24) | u8 |
| Quantile crossing flag (0/1) | u8 |
| Target, point median | 2 × f64 |
| Issued inverse-CDF quantiles (P01–P20 only) | 9 × f64 |
| CRPS, mean absolute support distance, half pairwise support distance (P01–P20 only) | 3 × f64 |
| Pinball contributions (P01–P20 only) | 9 × f64 |
| Four interval lower/upper pairs | 8 × f64 |

Record widths: 258 bytes for P01–P20, 90 for P21–P24. Quantile and coverage order defined in manifest.md. No missing-value sentinel inside binary records; conformal records omit distribution fields. The independent audit reconstructs support from archived parameters and original frozen source predictions, including calibration analog search; it does not rely on exported CRPS terms. It checks exact origin, lead, target, record counts, crossing and all scores. It validates parameter structure/counts but is not an independent refit of every estimator.

## Metrics

Headerless CSV: fold, recipe, group (all/lead/month), index (0 for all, 1–24 or 1–12), scalar count, MAE, RMSE, CRPS, nine pinball losses, then coverage/width/interval-score triples at each of four nominal levels. Conformal CRPS and nine pinballs are NA. Values rounded to eight decimal places. 888 rows per fold, 1,776 across both. Costs CSV has header and 24 rows per fold; issue_score_export_s includes scoring and compression and is not deployment latency.

Parameter files start with recipe,year,ID,calibration_rows,count followed by method-specific records. Floating-point parameters use round-trip scientific notation. Analog index is reconstructed from source hashes, prior-year calibration and fixed scalar/tie rules; not duplicated as an additional large artifact.

## Reproduction

Compile score.rs and audit.rs with Rust edition 2021, optimized, and execute each with JOINED_DIR OUTPUT_DIR YEAR. Verify C013 joined source hashes before either execution. Only years 2021 and 2022 allowed. Synthetic tests are separate from measured assessments.
