#!/usr/bin/env bash
set -euo pipefail
out=/tmp/c011-timing-recovered.csv
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export work
tail -n +2 results/C011/registry.csv | while IFS=, read -r campaign method fold model seed track dimension origins mae rmse nat nmae nrmse observed sha run job status rescore; do
  if [[ $fold == 2019 && $seed == 42 ]]; then
    file="results/C011/screen/$method-$model.txt"
  else
    file="results/C011/annual/$method-$model-$fold-$seed.txt"
  fi
  if ! grep -q 'Elapsed (wall clock) time (h:mm:ss or m:ss): ' "$file" || ! grep -q 'Maximum resident set size (kbytes): ' "$file"; then
    printf '%s\n' "$job"
  fi
done > "$work/jobs"
echo "Missing archive timing records: $(wc -l < "$work/jobs")"
recover() {
  local job="$1" log="$work/$1.log"
  gh api --allow-escape-sequences "repos/$GITHUB_REPOSITORY/actions/jobs/$job/logs" > "$log"
  awk -v job="$job" '
    /Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): / {sub(/^.*\): /,"");sub(/\r$/,"");n=split($0,a,":");wall=0;for(i=1;i<=n;i++)wall=60*wall+a[i];wc++}
    /Maximum resident set size \(kbytes\): / {sub(/^.*\): /,"");sub(/\r$/,"");rss=$0;rc++}
    /Exit status: 0[[:space:]]*$/ {ec++}
    END {if(wc!=1||rc!=1||ec!=1||wall<=0||rss!~/^[0-9]+$/||rss<=0)exit 1;printf "%s,%.2f,%s\n",job,wall,rss}
  ' "$log" > "$work/$job.csv"
  rm "$log"
}
export -f recover
xargs -r -P 6 -n 1 bash -euo pipefail -c 'recover "$1"' _ < "$work/jobs"
printf 'job_id,wall_s,max_rss_kib\n' > "$out"
while read -r job; do cat "$work/$job.csv"; done < "$work/jobs" >> "$out"
test "$(wc -l < "$out")" -eq "$(( $(wc -l < "$work/jobs") + 1 ))"
cat "$out"
