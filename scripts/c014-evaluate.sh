#!/usr/bin/env bash
set -euo pipefail
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
year="$1"
case "$year" in 2021|2022) ;; *) exit 2;; esac
root="results-run/C014/$year"
mkdir -p "$root/joined" "$root/out"
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/15239944097" > "$root/source-meta.json"
jq -e '.name=="c013-joined-sources" and .expired==false and .workflow_run.id==46089681073 and .workflow_run.head_sha=="d5f37c4cc7742875ca1631be9a4533210dede85a"' "$root/source-meta.json" >/dev/null
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/15239944097/zip" > "$root/source.zip"
unzip -tq "$root/source.zip" >/dev/null
unzip -q "$root/source.zip" -d "$root/joined"
awk '$1 ~ /^[0-9a-f]+$/ && length($1)==64 && $2 ~ /^202[012](\.csv|-origins\.txt)$/ {print}' results/C013/source-audit.txt > "$root/frozen.sha256"
test "$(wc -l < "$root/frozen.sha256")" -eq 6
cmp "$root/frozen.sha256" "$root/joined/SHA256SUMS"
(cd "$root/joined" && sha256sum -c ../frozen.sha256)
rustc --edition=2021 -O --test campaigns/014-probabilistic/probabilistic.rs -o "$root/tests"
"$root/tests"
rustc --edition=2021 -O --test campaigns/014-probabilistic/audit.rs -o "$root/audit-tests"
"$root/audit-tests"
rustc --edition=2021 -O campaigns/014-probabilistic/score.rs -o "$root/score"
rustc --edition=2021 -O campaigns/014-probabilistic/audit.rs -o "$root/audit"
/usr/bin/time -v "$root/score" "$root/joined" "$root/out" "$year" 2> "$root/out/score-resources.txt" || { cat "$root/out/score-resources.txt"; exit 1; }
cat "$root/out/score-resources.txt"
/usr/bin/time -v "$root/audit" "$root/joined" "$root/out" "$year" 2> "$root/out/audit-resources.txt" | tee "$root/out/audit.txt" || { cat "$root/out/audit-resources.txt"; exit 1; }
cat "$root/out/audit-resources.txt"
test "$(wc -l < "$root/out/metrics.csv")" -eq 888
test "$(wc -l < "$root/out/costs.csv")" -eq 25
test "$(find "$root/out" -name 'P*-forecasts.bin.gz' | wc -l)" -eq 24
test "$(find "$root/out" -name 'P*-parameters.csv' | wc -l)" -eq 24
grep ",all,0," "$root/out/metrics.csv"
cat "$root/out/costs.csv"
(cd "$root/out" && sha256sum * > SHA256SUMS && cat SHA256SUMS)
echo "c014-evaluation,$year,24,pass"
