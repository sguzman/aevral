#!/usr/bin/env bash
set -euo pipefail
: "${FOLD:?}" "${ID:?}"
root="${1:?verified joined input root}"
dir=results-run
mkdir -p "$dir"
row=$(awk -F, -v y="$FOLD" -v m="$ID" '$1==y&&$2==m{print;n++}END{if(n!=1)exit 1}' campaigns/015-tournament/combination-matrix.csv)
IFS=, read -r year id recipe members calibration <<< "$row"
printf '%s\n' "$row" > "$dir/request.csv"
git rev-parse HEAD > "$dir/code-revision.txt"
finish(){
  local rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    printf '%s,%s,error,runner-or-audit-exit-%s\n' "$year" "$id" "$rc" > "$dir/outcome.csv"
    for f in predictions.csv predictions.csv.gz distribution.csv distribution.csv.gz; do
      if [[ -f "$dir/$f" ]]; then mv "$dir/$f" "$dir/unverified-$f"; fi
    done
  fi
  cat "$dir/outcome.csv"
  (cd "$dir"; rm -f SHA256SUMS; sha256sum * > SHA256SUMS; cat SHA256SUMS)
  exit "$rc"
}
trap finish EXIT
(cd "$root/joined"; sha256sum -c SHA256SUMS)
(cd "$root/cohorts"; sha256sum -c SHA256SUMS)
case "$id" in E08|P13) k=4;; E13|E14) k=8;; *) exit 1;; esac
missing=()
if [[ ! -f "$root/joined/$year-$k.csv" ]]; then missing+=("$year"); fi
if [[ "$id" != E08 ]]; then
  for ((y=2020;y<year;y++)); do
    if ((y<=2022)); then file="$y.csv"; else file="$y-$k.csv"; fi
    if [[ ! -f "$root/joined/$file" ]]; then missing+=("$y"); fi
  done
fi
if (( ${#missing[@]} > 0 )); then
  printf '%s,%s,blocked,missing-required-members-%s\n' "$year" "$id" "$(IFS=+; echo "${missing[*]}")" > "$dir/outcome.csv"
  exit 0
fi
rustc --edition=2021 -O campaigns/015-tournament/combinations.rs -o /tmp/combinations
rustc --edition=2021 -O campaigns/015-tournament/audit-predictions.rs -o /tmp/audit
set +e
/usr/bin/time -v timeout --kill-after=10s 1400s /tmp/combinations "$root/joined" "$dir" "$year" "$id" 2>"$dir/resources.txt" | tee "$dir/metrics.txt"
rc=${PIPESTATUS[0]}
set -e
cat "$dir/resources.txt"
if [[ "$rc" == 124 || "$rc" == 137 ]]; then
  for f in predictions.csv distribution.csv; do if [[ -f "$dir/$f" ]]; then mv "$dir/$f" "$dir/blocked-$f"; fi; done
  printf '%s,%s,blocked,runtime-guard\n' "$year" "$id" > "$dir/outcome.csv"
  exit 0
fi
test "$rc" -eq 0
/tmp/audit "$root/cohorts/$year-matched-targets.csv" "$dir/predictions.csv" "$dir/metrics.txt" "$year" "$id" matched "$dir/independent-metrics.csv"
if [[ "$id" == P13 ]]; then
  rustc --edition=2021 -O campaigns/015-tournament/audit-p13.rs -o /tmp/p13-audit
  /usr/bin/time -v /tmp/p13-audit "$root/joined" "$dir" "$year" | tee "$dir/p13-audit.txt"
  # Distribution median must exactly match the point export on every key.
  awk -F, 'FNR==1{next} FILENAME==ARGV[1]{p[$3 SUBSEP $4]=$6;t[$3 SUBSEP $4]=$5;next} {k=$1 SUBSEP $2;if(!(k in p)||p[k]+0!=$4+0||t[k]+0!=$3+0)exit 1;delete p[k]} END{if(length(p))exit 1}' "$dir/predictions.csv" "$dir/distribution.csv"
  gzip -n -1 "$dir/distribution.csv"
fi
gzip -n "$dir/predictions.csv"
printf '%s,%s,measured,independent-audit-pass\n' "$year" "$id" > "$dir/outcome.csv"
