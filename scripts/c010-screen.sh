#!/usr/bin/env bash
set -euo pipefail
: "${METHOD:?}" "${CONFIG:?}" "${FOLD:?}" "${SEED:?}"
test "$FOLD" = 2019
test "$SEED" = 42
origins=7455
awk -F, -v m="$METHOD" -v c="$CONFIG" '$1==m && $2==c && $6==42 && $7==2019 && $8==8192 && $9==12 && $10==0.001 {n++} END{exit n!=1}' campaigns/010-forecast-strategy/screen-matrix.csv
mkdir -p results-run
dir=results-run
echo "strategy-start,$FOLD,$CONFIG,$SEED"
if /usr/bin/time -v target/release/aevral strategy data/raw/ksea "$FOLD" "$dir/predictions.csv" "$CONFIG" "$SEED" 2>"$dir/resources.txt" | tee "$dir/metrics.txt"; then
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
  test "$(grep -c "^strategy-config,$FOLD,$CONFIG,$SEED,8192,12," "$dir/metrics.txt")" -eq 1
  test "$(grep -c "^strategy-epoch,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 12
  test "$(grep -c "^strategy-fit,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 1
  if [[ "$CONFIG" == correction-* ]]; then
    test "$(grep -c "^strategy-base-epoch,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 12
    awk -F, -v f="$FOLD" -v m="$CONFIG" -v s="$SEED" '$1=="strategy-split" && $2==f && $3==m && $4==s {if($5<=0||$6!=2048||$5+$7!=6144||$8!=12*($5+$6)) exit 1; n++} END{exit n!=1}' "$dir/metrics.txt"
  fi
  if [[ "$CONFIG" == pinball-multi ]]; then
    awk -F, -v f="$FOLD" -v m="$CONFIG" -v s="$SEED" -v n="$origins" '$1=="strategy-quantile" && $2==f && $3==m && $4==s {if($5<0||$5>$6||$6!=24*n) exit 1; k++} END{exit k!=1}' "$dir/metrics.txt"
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
echo "strategy-end,$FOLD,$CONFIG,$SEED"
(cd results-run && sha256sum *.txt *.gz > SHA256SUMS)
