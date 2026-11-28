#!/usr/bin/env bash
set -euo pipefail
: "${FOLD:?}" "${ID:?}" "${SEED:?}"
matrix=campaigns/015-tournament/fit-matrix.csv
row=$(awk -F, -v y="$FOLD" -v m="$ID" -v s="$SEED" '$1==y&&$2==m&&$4==s{print;n++}END{if(n!=1)exit 1}' "$matrix")
IFS=, read -r year id recipe seed budget track backend role <<< "$row"
dir=results-run
cohort=cohort-run
mkdir -p "$dir" "$cohort"
test "$(wc -l < results/AEV-015/snapshot.sha256)" -eq 27
(cd data/raw/ksea && sha256sum -c ../../../results/AEV-015/snapshot.sha256)
# Reconstruct the cohort from the exact frozen source, then compare every archived
# digest. This avoids depending on artifact retention for cohort reconstruction.
target/release/aevral retrospective-cohort data/raw/ksea "$year" "$cohort" > "$cohort/audit.txt"
grep -Fxq "retrospective-cohort-audit,$year,pass" "$cohort/audit.txt"
awk -F, -v y="$year" '$1==y{print $3 "  " $2;n++}END{if(n!=12)exit 1}' results/C015/cohort-hashes.csv > "$dir/cohort.sha256"
(cd "$cohort" && sha256sum -c ../"$dir/cohort.sha256")
awk -F, -v y="$year" '$1==y{print $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8;n++}END{if(n!=1)exit 1}' results/C015/cohorts.csv | cmp - <(tail -n1 "$cohort/cohort.csv")
IFS=, read -r _ start end train matched natural possible complete < <(tail -n1 "$cohort/cohort.csv")
if [[ "$budget" == full ]]; then
  fit="$train"; fitfile=train-origins.txt
else
  fit="$budget"; fitfile="fit-$budget-origins.txt"
fi
printf '%s\n' "$row" > "$dir/request.csv"
git rev-parse HEAD > "$dir/code-revision.txt"
rustc --version > "$dir/compiler.txt"
if [[ "$backend" != native ]]; then
  target/release/aevral upstream-smoke "$backend" | tee "$dir/backend-smoke.txt"
  case "$backend" in
    lightgbm) test -x "$AEVRAL_LIGHTGBM"; [[ "$AEVRAL_LIGHTGBM" == *lightgbm-4.6.0* ]]; printf '%s\n' "$AEVRAL_LIGHTGBM" > "$dir/backend-path.txt";;
    xgboost) test -f "$AEVRAL_XGBOOST_LIB"; [[ "$AEVRAL_XGBOOST_LIB" == *xgboost-3.0.5* ]]; printf '%s\n' "$AEVRAL_XGBOOST_LIB" > "$dir/backend-path.txt";;
    *) exit 1;;
  esac
fi
rustc --edition=2021 -O campaigns/015-tournament/audit-predictions.rs -o /tmp/c015-audit
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "c015-preflight,$year,$id,$seed,pass"
  exit 0
fi
# Reserve remaining job time for audit and upload. Preserve stricter original
# in-model guards; outer limit covers loading and audit-independent overhead.
set +e
/usr/bin/time -v timeout --kill-after=10s 1500s target/release/aevral retrospective data/raw/ksea "$year" "$dir/predictions.csv" "$id" "$seed" 2>"$dir/resources.txt" | tee "$dir/metrics.txt"
rc=${PIPESTATUS[0]}
set -e
cat "$dir/resources.txt"
if [[ "$rc" == 124 || "$rc" == 137 ]]; then
  printf '%s,%s,%s,blocked,outer-runtime-limit\n' "$year" "$id" "$seed" > "$dir/outcome.csv"
elif [[ "$rc" != 0 ]]; then
  printf '%s,%s,%s,error,process-exit-%s\n' "$year" "$id" "$seed" "$rc" > "$dir/outcome.csv"
  exit "$rc"
else
  grep -Fxq "retrospective-request,$year,$id,$seed,$recipe,$fit,$track,$backend" "$dir/metrics.txt"
  grep -Fxq "retrospective-end,$year,$id,$seed" "$dir/metrics.txt"
  cmp "$cohort/train-origins.txt" "$dir/train-origins.txt"
  cmp "$cohort/matched-origins.txt" "$dir/matched-origins.txt"
  cmp "$cohort/$fitfile" "$dir/fit-origins.txt"
  if [[ "$id" == G || "$id" == K || "$id" == N ]]; then cmp "$cohort/$fitfile" "$dir/backend-fit-origins.txt"; fi
  if [[ "$id" == I ]]; then
    /tmp/c015-audit "$cohort/matched-targets.csv" "$dir/daily-predictions.csv" "$dir/metrics.txt" "$year" daily matched "$dir/daily-audit.csv"
    gzip "$dir/daily-predictions.csv"
  fi
  nscore=$(grep -c "^score,$year,$recipe," "$dir/metrics.txt" || true)
  nblock=$(grep -c "^blocked,$year,$recipe," "$dir/metrics.txt" || true)
  test "$((nscore+nblock))" -eq 1
  if [[ "$nscore" == 1 ]]; then
    /tmp/c015-audit "$cohort/matched-targets.csv" "$dir/predictions.csv" "$dir/metrics.txt" "$year" "$recipe" matched "$dir/independent-metrics.csv"
    if [[ "$id" == M || "$id" == S ]]; then
      awk -F, -v y="$year" -v r="$recipe" -v s="$seed" -v b="$fit" -v id="$id" '
        $1=="optimization-training"&&$2==y&&$3==r&&$4==s{
          if(NF!=12||$5<1||$5>30||$6<1||$6>$5||$10+$11+$12!=b||$8!=$5*$10||$9>$7)exit 1
          batch=id=="M"?32:1
          if($7!=$5*int(($10+batch-1)/batch))exit 1
          if(id=="M"&&($5!=30||$6!=30||$10!=b||$11!=0||$12!=0))exit 1
          if(id=="S"&&($11!=int((b+9)/10)||$10<1))exit 1
          print $10,$11,$12;n++
        }END{if(n!=1)exit 1}' "$dir/metrics.txt" > "$dir/fit-counts.txt"
      read -r actual val purged < "$dir/fit-counts.txt"
      head -n "$actual" "$dir/fit-origins.txt" | cmp - "$dir/actual-fit-origins.txt"
      test "$(wc -l < "$dir/internal-validation-origins.txt")" -eq "$val"
      if (( val > 0 )); then
        tail -n "$val" "$dir/fit-origins.txt" | cmp - "$dir/internal-validation-origins.txt"
        firstval=$(head -n1 "$dir/internal-validation-origins.txt")
        expected=$(awk -v t="$firstval" '$1+1440<=t{n++}END{print n+0}' "$dir/fit-origins.txt")
        test "$expected" -eq "$actual"
      fi
    fi
    if [[ "$id" == S ]]; then
      cmp "$cohort/natural-origins.txt" "$dir/natural-origins.txt"
      /tmp/c015-audit "$cohort/natural-targets.csv" "$dir/natural-predictions.csv" "$dir/metrics.txt" "$year" "$recipe" natural "$dir/natural-audit.csv"
      # Check overlap forecasts exactly, not merely matching scores.
      awk -F, 'FNR==1{next} FILENAME==ARGV[1]{p[$3 SUBSEP $4]=$6;next} {k=$3 SUBSEP $4;if(k in p){if(p[k]!=$6)exit 1;delete p[k]}} END{if(length(p))exit 1}' "$dir/predictions.csv" "$dir/natural-predictions.csv"
      gzip "$dir/natural-predictions.csv"
    fi
    if [[ "$id" == G || "$id" == L ]]; then test "$(grep -c "^upstream-lead,$backend," "$dir/metrics.txt")" -eq 24; fi
    if [[ "$id" == N ]]; then test "$(grep -c "^strategy-epoch,$year,$recipe,$seed," "$dir/metrics.txt")" -eq 12; fi
    if [[ "$id" == W ]]; then
      test "$(grep -c "^representation-epoch,$year,$recipe,$seed," "$dir/metrics.txt")" -eq 12
      awk -F, -v y="$year" -v r="$recipe" '
        FILENAME==ARGV[1]{if(FNR==1)next;splitname=$1=="matched"?"eval":"train";n[splitname]++;for(j=0;j<3;j++)seen[splitname,j]+=$(j+3);all[splitname]+=($3&&$4&&$5);next}
        $1=="representation-available"&&$2==y&&$3==r{if($5!=n[$4]||$6!=all[$4])exit 1;a++}
        $1=="representation-availability"&&$2==y&&$3==r{if($6!=seen[$4,$5]||$7!=n[$4]-$6)exit 1;b++}
        END{if(a!=2||b!=6)exit 1}' "$cohort/pressure-masks.csv" "$dir/metrics.txt"
    fi
    gzip "$dir/predictions.csv"
    printf '%s,%s,%s,measured,independent-audit-pass\n' "$year" "$id" "$seed" > "$dir/outcome.csv"
  else
    grep "^blocked,$year,$recipe," "$dir/metrics.txt" > "$dir/block-reason.txt"
    if [[ -f "$dir/predictions.csv" ]]; then
      test "$(wc -l < "$dir/predictions.csv")" -le 1
      mv "$dir/predictions.csv" "$dir/blocked-predictions.csv"
    fi
    printf '%s,%s,%s,blocked,model-resource-guard\n' "$year" "$id" "$seed" > "$dir/outcome.csv"
  fi
fi
# Never expose partial forecasts as measured output.
if [[ -f "$dir/predictions.csv" ]]; then mv "$dir/predictions.csv" "$dir/blocked-predictions.csv"; fi
cat "$dir/outcome.csv"
(cd "$dir"; sha256sum * > SHA256SUMS; sha256sum -c SHA256SUMS; cat SHA256SUMS)
