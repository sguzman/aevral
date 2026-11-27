#!/usr/bin/env bash
set -euo pipefail
fold="$1"
group="$2"
case "$group" in
  autoreg) configs=(ar-72 ar-48 arx-72 arx-48 var-1 varx-1) ;;
  arima) configs=(arima-101 arima-211 arimax-101 arimax-111 sarima-101-010 sarima-101-100) ;;
  smoothing) configs=(ets-aaa ets-ann harmonic-2-ar1 harmonic-1-ar1) ;;
  kalman) configs=(struct-season struct-level dlm-harmonic dlm-trend-harmonic) ;;
  switching) configs=(switch-ar6 switch-ar1 garch-ar24 garch-ar6) ;;
  *) echo "unknown finalist group" >&2; exit 2 ;;
esac
mkdir -p results-run
for label in "${configs[@]}"; do
  dir="results-run/$label"
  mkdir -p "$dir"
  echo "finalist-start,$fold,$label"
  /usr/bin/time -v target/release/aevral "$group" data/raw/ksea "$fold" "$dir/predictions.csv" "$label" 2> "$dir/resources.txt" | tee "$dir/metrics.txt"
  cat "$dir/resources.txt"
  gzip "$dir/predictions.csv"
  if test -f "$dir/predictions.variance.csv"; then gzip "$dir/predictions.variance.csv"; fi
  test "$(grep -Ec "^(score|blocked),$fold,$label," "$dir/metrics.txt")" -eq 1
  if grep -q "^score,$fold,$label," "$dir/metrics.txt"; then
    test "$(grep -c "^lead,$fold,$label," "$dir/metrics.txt")" -eq 24
    if [[ "$label" == garch-* ]]; then
      test "$(grep -c "^variance-score,$fold,$label," "$dir/metrics.txt")" -eq 1
      test "$(grep -c "^variance-lead,$fold,$label," "$dir/metrics.txt")" -eq 24
    fi
  fi
  echo "finalist-end,$fold,$label"
done
