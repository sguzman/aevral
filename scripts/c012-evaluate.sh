#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG:?}" "${FOLD:?}" "${SEED:?}" "${RUNG:?}"
case "$FOLD" in
  2019) train=121849; origins=7455;;
  2020) train=129327; origins=7543;;
  2021) train=136893; origins=7501;;
  2022) train=144417; origins=7639;;
  *) exit 1;;
esac
case "$RUNG" in
  1) budget=2048; rung_name=One;;
  2) budget=8192; rung_name=Two;;
  3) budget=32768; rung_name=Three;;
  final) budget=32768; rung_name=Final;;
  *) exit 1;;
esac
grid=campaigns/012-optimization/candidate-grid.csv
line=$(awk -F, -v id="$CONFIG" '$5==id{print;n++}END{if(n!=1)exit 1}' "$grid")
IFS=, read -r family model track seed_policy config recipe reference <<< "$line"
test "$config" = "$CONFIG"
if [[ "$RUNG" != final ]]; then test "$FOLD,$SEED" = "2019,42"; fi
if [[ "$seed_policy" == deterministic ]]; then test "$SEED" = 42; fi
if [[ "$RUNG" == final ]]; then test "$FOLD,$SEED" != "2019,42"; fi
case "$SEED" in 42|123|2026) ;; *) exit 1;; esac
grep -Fxq "$CONFIG,$FOLD,$SEED" "campaigns/012-optimization/rung-$RUNG.csv"
mkdir -p results-run
dir=results-run
echo "optimization-start,$FOLD,$CONFIG,$SEED,$RUNG"
if /usr/bin/time -v target/release/aevral optimization data/raw/ksea "$FOLD" "$dir/predictions.csv" "$CONFIG" "$RUNG" "$SEED" 2>"$dir/resources.txt" | tee "$dir/metrics.txt"; then
  cat "$dir/resources.txt"
else
  cat "$dir/resources.txt" >&2; exit 1
fi
grep -Eq 'Elapsed \(wall clock\) time' "$dir/resources.txt"
grep -Eq 'Maximum resident set size \(kbytes\): [0-9]+' "$dir/resources.txt"
grep -Eq 'Exit status: 0$' "$dir/resources.txt"
grep -Fxq "optimization-request,$FOLD,$CONFIG,$SEED,$rung_name,$budget,$track,$recipe" "$dir/metrics.txt"
test "$(grep -Ec "^(score|blocked),$FOLD,$CONFIG," "$dir/metrics.txt")" -eq 1
test "$(wc -l < "$dir/train-origins.txt")" -eq "$train"
test "$(wc -l < "$dir/matched-origins.txt")" -eq "$origins"
test "$(wc -l < "$dir/fit-origins.txt")" -eq "$budget"
IFS=, read -r year train_sha matched_sha fit_sha <<< "$(grep "^$FOLD," campaigns/012-optimization/cohort-hashes.csv)"
(cd "$dir"; printf '%s  train-origins.txt\n%s  matched-origins.txt\n' "$train_sha" "$matched_sha" | sha256sum -c -)
awk -v n="$train" -v b="$budget" 'BEGIN{i=0} NR==int(i*n/b)+1{print;i++}END{if(i!=b)exit 1}' "$dir/train-origins.txt" | cmp - "$dir/fit-origins.txt"
if [[ "$budget" == 8192 ]]; then (cd "$dir"; printf '%s  fit-origins.txt\n' "$fit_sha" | sha256sum -c -); fi
if grep -q "^score,$FOLD,$CONFIG," "$dir/metrics.txt"; then
  test "$(grep -c "^optimization-cost,$FOLD,$CONFIG," "$dir/metrics.txt")" -eq 1
  if [[ "$seed_policy" == stochastic ]]; then
    test "$(grep -c "^optimization-training,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 1
    awk -F, -v n="$budget" -v recipe="$recipe" '
    /^optimization-training,/{
      if(NF!=12||$5<1||$6<1||$6>$5||$10+$11+$12!=n||$8!=$5*$10)exit 1
      b=recipe~/b32/?32:1;epochs=recipe~/e12/?12:30
      if($7!=$5*int(($10+b-1)/b)||$9>$7||$5>epochs)exit 1
      if(recipe!~/stop/&&($5!=epochs||$6!=$5||$10!=n||$11!=0||$12!=0))exit 1
      if(recipe~/stop/&&($11!=int((n+9)/10)||$10<1))exit 1
      print $10,$11
    }' "$dir/metrics.txt" > "$dir/fit-counts.txt"
    read -r fit val < "$dir/fit-counts.txt"
    head -n "$fit" "$dir/fit-origins.txt" | cmp - "$dir/actual-fit-origins.txt"
    test "$(wc -l < "$dir/internal-validation-origins.txt")" -eq "$val"
    if (( val > 0 )); then
      tail -n "$val" "$dir/fit-origins.txt" | cmp - "$dir/internal-validation-origins.txt"
      test "$(( $(tail -1 "$dir/actual-fit-origins.txt") + 1440 ))" -le "$(head -1 "$dir/internal-validation-origins.txt")"
    fi
  fi
  # Independent export check: exact origin list, 24 leads, finite values, independently pooled and per-lead MAE/RMSE.
  verify_csv(){
    local file="$1" originfile="$2" prefix="$3"
    awk -F, -v f="$FOLD" -v m="$CONFIG" -v tag="$prefix" '
      FILENAME==ARGV[1] { origins[++n]=$1; next }
      FILENAME==ARGV[2] {
        if($1==tag"score"&&$2==f&&$3==m){expected=$4;mae=$5;rmse=$6;scores++}
        if($1==tag"lead"&&$2==f&&$3==m){if($4<1||$4>24||seen[$4]++)exit 1;la[$4]=$5;lr[$4]=$6;leads++}
        if($1==tag"month"&&$2==f&&$3==m){if($4<1||$4>12||ms[$4]++)exit 1;mc+=$5;months++}
        next
      }
      FNR==1 {if($0!="fold,model,origin_utc_minutes_since_2000,lead,target_c,prediction_c")exit 1;next}
      {
        lead=(FNR-2)%24+1; i=int((FNR-2)/24)+1
        if(NF!=6||$1!=f||$2!=m||$3!=origins[i]||$4!=lead||i>n)exit 1
        if($5!~/^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/||$6!~/^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/)exit 1
        e=$6-$5;a=e<0?-e:e;aa+=a;ss+=e*e;al[lead]+=a;sl[lead]+=e*e;count++
      }
      function abs(x){return x<0?-x:x}
      END {
        if(scores!=1||expected!=n||n==0||count!=24*n||leads!=24||months!=12||mc!=n)exit 1
        if(abs(aa/count-mae)>1e-7||abs(sqrt(ss/count)-rmse)>1e-7)exit 1
        for(j=1;j<=24;j++)if(abs(al[j]/n-la[j])>1e-7||abs(sqrt(sl[j]/n)-lr[j])>1e-7)exit 1
        printf "optimization-rescore,%s,%s,%s,%d,%.8f,%.8f,pass\n",f,m,tag==""?"matched":"natural",n,aa/count,sqrt(ss/count)
      }' "$originfile" "$dir/metrics.txt" "$file" | tee -a "$dir/rescore.txt"
  }
  verify_csv "$dir/predictions.csv" "$dir/matched-origins.txt" ""
  if [[ "$family" == O02 ]]; then
    verify_csv "$dir/natural-predictions.csv" "$dir/natural-origins.txt" "natural-"
    awk -F, -v f="$FOLD" -v m="$CONFIG" -v expected="$origins" '$1=="optimization-natural"&&$2==f&&$3==m{if($4<$5||$5<expected||$6!=expected)exit 1;n++}END{exit n!=1}' "$dir/metrics.txt"
    gzip "$dir/natural-predictions.csv"
  else
    grep -Eq "^optimization-natural,$FOLD,$CONFIG,[0-9]+,$origins,$origins$" "$dir/metrics.txt"
  fi
  gzip "$dir/predictions.csv"
else
  if [[ -f "$dir/predictions.csv" ]]; then
    test "$(wc -l < "$dir/predictions.csv")" -le 1
    mv "$dir/predictions.csv" "$dir/blocked-predictions.csv"
  fi
fi
grep -Fxq "optimization-end,$FOLD,$CONFIG,$SEED,$rung_name" "$dir/metrics.txt"
echo "optimization-audit,$FOLD,$CONFIG,$SEED,$RUNG,pass"
(cd "$dir" && sha256sum * > SHA256SUMS && cat SHA256SUMS)
