#!/usr/bin/env bash
set -euo pipefail
# A single frozen recipe, annual fold and seed per job.
: "${METHOD:?}" "${CONFIG:?}" "${FOLD:?}" "${SEED:?}"
matrix=campaigns/008-decomposition-reservoir/finalist-matrix.csv
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
  D01|D02) cmd=decomp ;;
  D03|D04|D05) cmd=dyn ;;
  D06|D07) cmd=random-learner; args=("$SEED"); case "$SEED" in 42|123|2026) ;; *) exit 2 ;; esac ;;
  D08|D09) cmd=alternative ;;
  *) exit 2 ;;
esac
if [[ "$METHOD" != D06 && "$METHOD" != D07 ]]; then
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
  if [[ "$METHOD" == D06 || "$METHOD" == D07 ]]; then
    test "$(grep -c "^random-config,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 1
  fi
  # Validate individual prediction rows: sorted contiguous leads, unique origins,
  # exact model/fold, finite Celsius values, and exact number of origins.
  awk -F, -v f="$FOLD" -v m="$CONFIG" -v n="$origins" '
    NR==1 { if ($0 != "fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c") exit 1; next }
    {
      if (NF!=6 || $1!=f || $2!=m || $3 !~ /^-?[0-9]+$/ || $4 != ((NR-2)%24)+1) exit 1
      if ($5 !~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ || $6 !~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/) exit 1
      if ($4==1) { if (count>0 && $3+0<=prev+0) exit 1; prev=$3; count++ }
      else if ($3!=prev) exit 1
    }
    END { if (count!=n || NR!=n*24+1) exit 1 }
  ' "$dir/predictions.csv"
  awk -F, -v f="$FOLD" -v m="$CONFIG" -v n="$origins" '
    $1=="lead" && $2==f && $3==m { if ($4<1 || $4>24 || seen[$4]++) exit 1; leads++ }
    $1=="month" && $2==f && $3==m { if (months_seen[$4]++) exit 1; months++; total+=$5 }
    END { if (leads!=24 || months!=12 || total!=n) exit 1 }
  ' "$dir/metrics.txt"
  gzip "$dir/predictions.csv"
else
  # Blocked predictions may be partial; retain as a separately named diagnostic.
  mv "$dir/predictions.csv" "$dir/blocked-predictions.csv"
  gzip "$dir/blocked-predictions.csv"
fi
echo "finalist-end,$FOLD,$CONFIG,$SEED"
