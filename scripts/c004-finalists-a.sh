#!/usr/bin/env bash
set -euo pipefail
fold="$1"
group="$2"
case "$group" in
  linear) configs=('ols|ols' 'ridge-1e-4|ridge-1e-4' 'ridge-1e-6|ridge-1e-6') ;;
  sparse) configs=('lasso-1e-4|lasso-1e-4' 'lasso-1e-3|lasso-1e-3' 'enet-1e-3-a25|enet-1e-3-a25' 'enet-1e-3-a75|enet-1e-3-a75') ;;
  robust) configs=('huber-0.25|huber-0.25' 'huber-0.5|huber-0.5' 'median-1e-4|median-1e-4' 'median-0|median-0') ;;
  latent) configs=('pcr-64|pcr-64' 'pcr-32|pcr-32' 'pls-32|pls-32' 'pls-16|pls-16' 'rrr-16|rrr-16' 'rrr-8|rrr-8') ;;
  basis) configs=('gam-7-1e-3|gam-7:1e-3' 'gam-3-1e-3|gam-3:1e-3') ;;
  mars) configs=('mars-grid-2-64|mars-grid-2-64' 'mars-grid-1-64|mars-grid-1-64') ;;
  *) echo "unknown finalist group" >&2; exit 2 ;;
esac
mkdir -p results-run
for entry in "${configs[@]}"; do
  label="${entry%%|*}"
  arg="${entry#*|}"
  dir="results-run/$label"
  mkdir -p "$dir"
  /usr/bin/time -v target/release/aevral "$group" data/raw/ksea "$fold" "$dir/predictions.csv" "$arg" 2> "$dir/resources.txt" | tee "$dir/metrics.txt"
  cat "$dir/resources.txt"
  gzip "$dir/predictions.csv"
  grep -E "^(score|blocked),$fold,$label," "$dir/metrics.txt"
done
