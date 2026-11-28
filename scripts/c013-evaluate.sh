#!/usr/bin/env bash
# Frozen C013 scoring; run only after engine/auditor compile gate succeeds.
set -euo pipefail
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
root=results-run/C013
mkdir -p "$root/joined" "$root/out"
artifact=15239944097
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact" > "$root/metadata.json"
jq -e '.name=="c013-joined-sources" and .expired==false and .workflow_run.id==46089681073 and .workflow_run.head_sha=="d5f37c4cc7742875ca1631be9a4533210dede85a"' "$root/metadata.json" >/dev/null
gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact/zip" > "$root/source.zip"
unzip -tq "$root/source.zip" >/dev/null
unzip -q "$root/source.zip" -d "$root/joined"
awk '$1 ~ /^[0-9a-f]+$/ && length($1)==64 && $2 ~ /^202[012](\.csv|-origins\.txt)$/ {print}' results/C013/source-audit.txt > "$root/frozen.sha256"
test "$(wc -l < "$root/frozen.sha256")" -eq 6
cmp "$root/frozen.sha256" "$root/joined/SHA256SUMS"
(cd "$root/joined" && sha256sum -c ../frozen.sha256)
rustc --edition=2021 --test campaigns/013-ensembles/ensemble.rs -o "$root/tests"
"$root/tests"
rustc --edition=2021 -O campaigns/013-ensembles/score-ensembles.rs -o "$root/score"
rustc --edition=2021 -O campaigns/013-ensembles/audit-exports.rs -o "$root/audit"
/usr/bin/time -v "$root/score" "$root/joined" "$root/out" 2> "$root/out/resources.txt"
cat "$root/out/resources.txt"
"$root/audit" "$root/joined" "$root/out" | tee "$root/out/audit.txt"
test "$(grep -c '^score,.*,[E][0-9][0-9],' "$root/out/metrics.csv")" -eq 48
test "$(wc -l < "$root/out/metrics.csv")" -eq 2442
test "$(wc -l < "$root/out/coefficients.csv")" -eq 263
test "$(wc -l < "$root/out/parameters.csv")" -eq 228
# Print bounded evidence in the log; full metrics and predictions remain in artifact.
grep '^score,' "$root/out/metrics.csv"
grep -E '^(calibration|selected|spread-threshold),' "$root/out/parameters.csv"
gzip -n "$root/out"/predictions-*.csv
(cd "$root/out" && sha256sum * > SHA256SUMS && cat SHA256SUMS)
echo 'c013-evaluation,48,pass'
