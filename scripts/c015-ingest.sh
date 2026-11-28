#!/usr/bin/env bash
# Materialize verified C015 fit exports from a completed approved run.
# A failed or absent fit is explicitly unresolved, never treated as measured.
set -euo pipefail
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
run="${1:?fit run id}"
root="${2:?output directory}"
[[ "$run" =~ ^[0-9]+$ ]]
mkdir -p "$root"/{joined,cohorts,exports,source,registry}
gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run" > "$root/source/run.json"
jq -e '.name=="C015 frozen single-model tournament" and .status=="completed" and .head_branch=="main"' "$root/source/run.json" >/dev/null
sha=$(jq -r .head_sha "$root/source/run.json")
gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs/$run/artifacts?per_page=100" | jq -s '[.[].artifacts[]]' > "$root/source/artifacts.json"
gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs/$run/jobs?per_page=100" | jq -s '[.[].jobs[]]' > "$root/source/jobs.json"
# The producer run must contain precisely the frozen identity matrix.
test "$(jq '[.[]|select(.name|startswith("fit ("))]|length' "$root/source/jobs.json")" -eq 80
matrix=campaigns/015-tournament/fit-matrix.csv
test "$(git hash-object "$matrix")" = b2ac045702bd22f9f12b32b9b96f4677e3c5b6cb
rustc --edition=2021 -O campaigns/015-tournament/audit-predictions.rs -o "$root/audit"
rustc --edition=2021 -O campaigns/015-tournament/join.rs -o "$root/join"
download(){
  local id="$1" name="$2" dest="$3" origin="$4" rev="$5"
  mkdir -p "$dest"
  gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$id" > "$dest.meta.json"
  jq -e --arg n "$name" --argjson r "$origin" --arg s "$rev" '.name==$n and .expired==false and .workflow_run.id==$r and .workflow_run.head_sha==$s' "$dest.meta.json" >/dev/null
  gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$id/zip" > "$dest.zip"
  unzip -tq "$dest.zip" >/dev/null
  unzip -q "$dest.zip" -d "$dest"
  rm "$dest.zip"
}
# Frozen historical C013 joins (only 2020..2022). Require exact six hashes.
download 15239944097 c013-joined-sources "$root/source/history" 46089681073 d5f37c4cc7742875ca1631be9a4533210dede85a
awk '$1 ~ /^[0-9a-f]+$/ && length($1)==64 && $2 ~ /^202[012](\.csv|-origins\.txt)$/ {print}' results/C013/source-audit.txt > "$root/source/history.sha256"
test "$(wc -l < "$root/source/history.sha256")" -eq 6
cmp "$root/source/history.sha256" "$root/source/history/SHA256SUMS"
(cd "$root/source/history"; sha256sum -c ../history.sha256)
for y in 2020 2021 2022; do cp "$root/source/history/$y.csv" "$root/joined/$y.csv"; done
# Cohort artifacts are independently frozen in Git; fail rather than replace
# any target or origin list. Their original workflow identity is fixed.
while IFS=, read -r y start end train matched natural possible complete quarantines cohortrun job artifact; do
  [[ "$y" == fold ]] && continue
  name=$(gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact" --jq .name)
  download "$artifact" "$name" "$root/cohorts/$y" "$cohortrun" 7cb963e94b60d9e0cb8d3a30613e8476983aa789
  awk -F, -v y="$y" '$1==y{print $3 "  " $2;n++}END{if(n!=12)exit 1}' results/C015/cohort-hashes.csv > "$root/cohorts/$y.sha256"
  (cd "$root/cohorts/$y"; sha256sum -c "../$y.sha256")
done < results/C015/cohorts.csv
printf 'fold,id,seed,recipe,status,reason,run,job,artifact,commit\n' > "$root/registry/fits.csv"
while IFS=, read -r y id recipe seed budget track backend role; do
  [[ "$y" == fold ]] && continue
  key="$y-$id-$seed"
  name="c015-fit-$key"
  jobname="fit ($y, $id, $seed)"
  job=$(jq -cer --arg n "$jobname" '[.[]|select(.name==$n)]|if length==1 then .[0] else error("job identity") end' "$root/source/jobs.json")
  jid=$(jq -r .id <<< "$job")
  conclusion=$(jq -r .conclusion <<< "$job")
  found=$(jq -cer --arg n "$name" '[.[]|select(.name==$n and .expired==false)]|if length>1 then error("duplicate artifact") else . end' "$root/source/artifacts.json")
  if [[ "$(jq length <<< "$found")" == 0 ]]; then
    printf '%s,%s,%s,%s,error,missing-artifact,%s,%s,NA,%s\n' "$y" "$id" "$seed" "$recipe" "$run" "$jid" "$sha" >> "$root/registry/fits.csv"
    continue
  fi
  aid=$(jq -r '.[0].id' <<< "$found")
  dest="$root/source/$key"
  download "$aid" "$name" "$dest" "$run" "$sha"
  if [[ "$conclusion" != success ]]; then
    printf '%s,%s,%s,%s,error,job-%s,%s,%s,%s,%s\n' "$y" "$id" "$seed" "$recipe" "$conclusion" "$run" "$jid" "$aid" "$sha" >> "$root/registry/fits.csv"
    continue
  fi
  test -f "$dest/SHA256SUMS"
  (cd "$dest"; sha256sum -c SHA256SUMS)
  grep -Fxq "$y,$id,$recipe,$seed,$budget,$track,$backend,$role" "$dest/request.csv"
  grep -Fxq "$sha" "$dest/code-revision.txt"
  IFS=, read -r oy oi os status reason < "$dest/outcome.csv"
  test "$oy,$oi,$os" = "$y,$id,$seed"
  [[ "$status" == measured || "$status" == blocked ]]
  if [[ "$id" == I && -f "$dest/daily-predictions.csv.gz" ]]; then
    gzip -dc "$dest/daily-predictions.csv.gz" > "$root/exports/$y-daily.csv"
    "$root/audit" "$root/cohorts/$y/matched-targets.csv" "$root/exports/$y-daily.csv" "$dest/metrics.txt" "$y" daily matched "$root/registry/$y-daily.csv"
    cmp "$root/registry/$y-daily.csv" "$dest/daily-audit.csv"
  fi
  if [[ "$status" == measured ]]; then
    grep -Fxq "$y,$id,$seed,measured,independent-audit-pass" "$dest/outcome.csv"
    gzip -dc "$dest/predictions.csv.gz" > "$root/exports/$key.csv"
    "$root/audit" "$root/cohorts/$y/matched-targets.csv" "$root/exports/$key.csv" "$dest/metrics.txt" "$y" "$recipe" matched "$root/registry/$key.csv"
    cmp "$root/registry/$key.csv" "$dest/independent-metrics.csv"
    bash scripts/c015-cost.sh "$y" "$id" "$recipe" "$seed" "$dest" > "$root/registry/$key-cost.csv"
    if [[ "$id" == S ]]; then
      gzip -dc "$dest/natural-predictions.csv.gz" > "$root/exports/$key-natural.csv"
      "$root/audit" "$root/cohorts/$y/natural-targets.csv" "$root/exports/$key-natural.csv" "$dest/metrics.txt" "$y" "$recipe" natural "$root/registry/$key-natural.csv"
      cmp "$root/registry/$key-natural.csv" "$dest/natural-audit.csv"
    fi
    if [[ "$seed" == 42 ]]; then ln "$root/exports/$key.csv" "$root/exports/$y-$id.csv"; fi
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$y" "$id" "$seed" "$recipe" "$status" "$reason" "$run" "$jid" "$aid" "$sha" >> "$root/registry/fits.csv"
done < "$matrix"
test "$(wc -l < "$root/registry/fits.csv")" -eq 81
# Only complete required member groups can be joined. A missing secondary
# member does not prevent a valid four-member group.
for y in 2023 2024 2025 2026; do
  n=$(awk -F, -v y="$y" '$1==y{print $5}' results/C015/cohorts.csv)
  members="$root/source/members-$y";mkdir -p "$members"
  for id in daily I L R M S A H X; do
    if [[ -f "$root/exports/$y-$id.csv" ]]; then ln "$root/exports/$y-$id.csv" "$members/$id.csv"; fi
  done
  if [[ -f "$members/daily.csv" && -f "$members/I.csv" && -f "$members/L.csv" && -f "$members/R.csv" && -f "$members/M.csv" ]]; then
    "$root/join" "$y" four "$root/cohorts/$y/matched-targets.csv" "$members" "$n" "$root/joined/$y-4.csv"
    if [[ -f "$members/S.csv" && -f "$members/A.csv" && -f "$members/H.csv" && -f "$members/X.csv" ]]; then
      "$root/join" "$y" eight "$root/cohorts/$y/matched-targets.csv" "$members" "$n" "$root/joined/$y-8.csv"
    fi
  fi
done
(cd "$root/joined"; sha256sum *.csv > SHA256SUMS; sha256sum -c SHA256SUMS)
cat "$root/registry/fits.csv"
echo "c015-ingest,$run,80,pass"
