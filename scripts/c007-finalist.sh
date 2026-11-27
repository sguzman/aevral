#!/usr/bin/env bash
set -euo pipefail
# A single frozen recipe, annual fold and seed per job.
: "${METHOD:?}" "${CONFIG:?}" "${FOLD:?}" "${SEED:?}"
matrix=campaigns/007-kernels-neighbors/finalist-matrix.csv
grep -Fxq "$METHOD,$CONFIG,$FOLD,$SEED" "$matrix"
case "$FOLD" in
  2019) origins=7455 ;;
  2020) origins=7543 ;;
  2021) origins=7501 ;;
  2022) origins=7639 ;;
  *) exit 2 ;;
esac
args=()
case "$METHOD" in
  K01|K02|K03) cmd=neighbors ;;
  K04) cmd=local ;;
  K05|K06) cmd=krr ;;
  K07) cmd=svr ;;
  K08|K09) cmd=gp ;;
  K10) cmd=kfeat; args=("$SEED"); case "$SEED" in 42|123|2026) ;; *) exit 2 ;; esac ;;
  K11) cmd=kfeat; args=(42) ;;
  *) exit 2 ;;
esac
if [[ "$METHOD" != K10 ]]; then
  test "$SEED" = deterministic
  test "$FOLD" != 2019
elif [[ "$FOLD" == 2019 ]]; then
  test "$SEED" != 42
fi
mkdir -p results-run
dir=results-run
echo "finalist-start,$FOLD,$CONFIG,$SEED"
if /usr/bin/time -v target/release/aevral "$cmd" data/raw/ksea "$FOLD" "$dir/predictions.csv" "$CONFIG" "${args[@]}" 2>"$dir/resources.txt" | tee "$dir/metrics.txt"; then
  cat "$dir/resources.txt"
else
  cat "$dir/resources.txt" >&2
  exit 1
fi
test "$(grep -Ec "^(score|blocked),$FOLD,$CONFIG," "$dir/metrics.txt")" -eq 1
if grep -q "^score,$FOLD,$CONFIG," "$dir/metrics.txt"; then
  test "$(grep -c "^score,$FOLD,$CONFIG,$origins," "$dir/metrics.txt")" -eq 1
  test "$(grep -c "^lead,$FOLD,$CONFIG," "$dir/metrics.txt")" -eq 24
  test "$(grep -c "^month,$FOLD,$CONFIG," "$dir/metrics.txt")" -eq 12
  test "$(wc -l < "$dir/predictions.csv")" -eq "$((origins * 24 + 1))"
  if [[ "$METHOD" == K07 ]]; then
    test "$(grep -c '^svr-lead,' "$dir/metrics.txt")" -eq 24
  fi
  gzip "$dir/predictions.csv"
else
  # Blocked predictions may be partial; retain as a separately named diagnostic.
  mv "$dir/predictions.csv" "$dir/blocked-predictions.csv"
  gzip "$dir/blocked-predictions.csv"
fi
echo "finalist-end,$FOLD,$CONFIG,$SEED"
