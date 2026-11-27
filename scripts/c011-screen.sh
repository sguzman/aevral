#!/usr/bin/env bash
set -euo pipefail
: "${METHOD:?}" "${CONFIG:?}" "${FOLD:?}" "${SEED:?}" "${TRACK:?}" "${DIMENSION:?}"
test "$FOLD" = 2019
test "$SEED" = 42
grep -Fxq "$METHOD,$CONFIG,$SEED,$FOLD,$TRACK,$DIMENSION" campaigns/011-feature-engineering/screen-matrix.csv
mkdir -p results-run
dir=results-run
echo "representation-start,$FOLD,$CONFIG,$SEED"
if /usr/bin/time -v target/release/aevral representation data/raw/ksea "$FOLD" "$dir/predictions.csv" "$CONFIG" "$SEED" 2>"$dir/resources.txt" | tee "$dir/metrics.txt"; then
  cat "$dir/resources.txt"
else
  cat "$dir/resources.txt" >&2
  exit 1
fi
test "$(grep -Ec "^(score|blocked),$FOLD,$CONFIG," "$dir/metrics.txt")" -eq 1
test "$(wc -l < "$dir/train-origins.txt")" -eq 121849
test "$(wc -l < "$dir/matched-origins.txt")" -eq 7455
test "$(wc -l < "$dir/fit-origins.txt")" -eq 8192
for f in train matched fit; do
  awk 'NR>1&&$1+0<=p{exit 1}{p=$1+0}' "$dir/$f-origins.txt"
done
if grep -q "^score,$FOLD,$CONFIG," "$dir/metrics.txt"; then
  test "$(grep -c "^representation-config,$FOLD,$CONFIG,$SEED,8192,12,0.001,$DIMENSION," "$dir/metrics.txt")" -eq 1
  test "$(grep -c "^representation-epoch,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 12
  test "$(grep -c "^representation-fit,$FOLD,$CONFIG,$SEED," "$dir/metrics.txt")" -eq 1
  if [[ "$CONFIG" == base80 ]]; then grep -Fxq "representation-control,$FOLD,$SEED,pass" "$dir/metrics.txt"; fi
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
        printf "representation-rescore,%s,%s,%s,%d,%.8f,%.8f,pass\n",f,m,tag==""?"matched":"natural",n,aa/count,sqrt(ss/count)
      }' "$originfile" "$dir/metrics.txt" "$file" | tee -a "$dir/rescore.txt"
  }
  verify_csv "$dir/predictions.csv" "$dir/matched-origins.txt" ""
  if [[ "$CONFIG" == suffix* ]]; then
    verify_csv "$dir/natural-predictions.csv" "$dir/natural-origins.txt" "natural-"
    awk -F, -v f="$FOLD" -v m="$CONFIG" '$1=="representation-natural"&&$2==f&&$3==m{if($4<$5||$5<7455||$6!=7455)exit 1;n++}END{exit n!=1}' "$dir/metrics.txt"
    gzip "$dir/natural-predictions.csv"
  else
    grep -Eq "^representation-natural,$FOLD,$CONFIG,[0-9]+,7455,7455$" "$dir/metrics.txt"
  fi
  gzip "$dir/predictions.csv"
else
  if [[ -f "$dir/predictions.csv" ]]; then mv "$dir/predictions.csv" "$dir/blocked-predictions.csv"; gzip "$dir/blocked-predictions.csv"; fi
fi
echo "representation-end,$FOLD,$CONFIG,$SEED"
(cd "$dir" && sha256sum * > SHA256SUMS && cat SHA256SUMS)
