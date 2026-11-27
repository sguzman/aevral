#!/usr/bin/env bash
set -euo pipefail
fold="$1"
group="$2"
seed="$3"
case "$group" in
  cart) configs=(cart-shared-d8-l32 cart-shared-d8-l128 cart-lead-d8-l32 cart-lead-d4-l32) ;;
  boosting) configs=(hist-64-d5 hist-32-d5 median-boost-64 median-boost-32) ;;
  forest) configs=(bag-32 bag-16 rf-32-27 rf-16-27 extra-32-27 extra-16-27) ;;
  upstream) configs=(xgboost-128 xgboost-64 lightgbm-128 lightgbm-64) ;;
  catboost) configs=(catboost-128 catboost-64) ;;
  *) echo "unknown group" >&2; exit 2 ;;
esac
case "$fold" in 2019|2020|2021|2022) ;; *) exit 2 ;; esac
case "$group" in
  forest|catboost) case "$seed" in 42|123|2026) ;; *) exit 2 ;; esac
    if [[ "$fold" == 2019 && "$seed" == 42 ]]; then exit 2; fi ;;
  *) if [[ "$seed" != deterministic || "$fold" == 2019 ]]; then exit 2; fi ;;
esac
mkdir -p results-run
for label in "${configs[@]}"; do
  dir="results-run/$label"
  mkdir -p "$dir"
  cmd="$group"
  args=()
  case "$group" in
    forest) args=("$seed") ;;
    catboost) cmd=upstream; args=("$seed") ;;
    upstream) args=(42) ;;
  esac
  echo "finalist-start,$fold,$label,$seed"
  /usr/bin/time -v target/release/aevral "$cmd" data/raw/ksea "$fold" "$dir/predictions.csv" "$label" "${args[@]}" 2> "$dir/resources.txt" | tee "$dir/metrics.txt"
  cat "$dir/resources.txt"
  gzip "$dir/predictions.csv"
  test "$(grep -Ec "^(score|blocked),$fold,$label," "$dir/metrics.txt")" -eq 1
  if grep -q "^score,$fold,$label," "$dir/metrics.txt"; then
    test "$(grep -c "^lead,$fold,$label," "$dir/metrics.txt")" -eq 24
    if [[ "$cmd" == upstream ]]; then test "$(grep -c '^upstream-lead,' "$dir/metrics.txt")" -eq 24; fi
  fi
  echo "finalist-end,$fold,$label,$seed"
done
