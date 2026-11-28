#!/usr/bin/env bash
# Re-ingest audited fits, validate the 16 combination artifacts, and compute
# final durable numerical tables without fitting any model.
set -euo pipefail
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
fitrun="${1:?fit run}"
combrun="${2:?combination run}"
root="${3:?workspace}"
[[ "$fitrun" =~ ^[0-9]+$ && "$combrun" =~ ^[0-9]+$ ]]
bash scripts/c015-ingest.sh "$fitrun" "$root/fit"
mkdir -p "$root"/{comb,final}
gh api "repos/$GITHUB_REPOSITORY/actions/runs/$combrun" > "$root/comb/run.json"
jq -e '.name=="C015 frozen combinations" and .status=="completed" and .head_branch=="main"' "$root/comb/run.json" >/dev/null
sha=$(jq -r .head_sha "$root/comb/run.json")
# The selected combination run must consume this exact selected fit run.
gh api "repos/$GITHUB_REPOSITORY/contents/campaigns/015-tournament/combinations-launch.txt?ref=$sha" --jq .content | base64 -d | grep -Fxq "$fitrun"
gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs/$combrun/artifacts?per_page=100" | jq -s '[.[].artifacts[]]' > "$root/comb/artifacts.json"
gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs/$combrun/jobs?per_page=100" | jq -s '[.[].jobs[]]' > "$root/comb/jobs.json"
test "$(jq '[.[]|select(.name|startswith("combination ("))]|length' "$root/comb/jobs.json")" -eq 16
rustc --edition=2021 -O campaigns/015-tournament/audit-p13.rs -o "$root/p13-audit"
rustc --edition=2021 -O campaigns/015-tournament/rank.rs -o "$root/rank"
rustc --edition=2021 -O campaigns/015-tournament/bootstrap.rs -o "$root/bootstrap"
printf 'fold,id,status,reason,run,job,artifact,commit\n' > "$root/final/combinations.csv"
while IFS=, read -r y id recipe members calibration; do
  [[ "$y" == fold ]] && continue
  key="$y-$id"
  name="c015-combination-$key"
  job=$(jq -cer --arg n "combination ($y, $id)" '[.[]|select(.name==$n)]|if length==1 then .[0] else error("job identity") end' "$root/comb/jobs.json")
  jid=$(jq -r .id <<< "$job")
  conclusion=$(jq -r .conclusion <<< "$job")
  found=$(jq -cer --arg n "$name" '[.[]|select(.name==$n and .expired==false)]|if length>1 then error("duplicate artifact") else . end' "$root/comb/artifacts.json")
  if [[ "$(jq length <<< "$found")" == 0 ]]; then
    printf '%s,%s,error,missing-artifact,%s,%s,NA,%s\n' "$y" "$id" "$combrun" "$jid" "$sha" >> "$root/final/combinations.csv"
    continue
  fi
  aid=$(jq -r '.[0].id' <<< "$found")
  dest="$root/comb/$key";mkdir -p "$dest"
  gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$aid" > "$dest.meta.json"
  jq -e --arg n "$name" --argjson r "$combrun" --arg s "$sha" '.name==$n and .expired==false and .workflow_run.id==$r and .workflow_run.head_sha==$s' "$dest.meta.json" >/dev/null
  gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$aid/zip" > "$dest.zip"
  unzip -tq "$dest.zip" >/dev/null;unzip -q "$dest.zip" -d "$dest";rm "$dest.zip"
  if [[ "$conclusion" != success ]]; then
    printf '%s,%s,error,job-%s,%s,%s,%s,%s\n' "$y" "$id" "$conclusion" "$combrun" "$jid" "$aid" "$sha" >> "$root/final/combinations.csv"
    continue
  fi
  (cd "$dest";sha256sum -c SHA256SUMS)
  grep -Fxq "$y,$id,$recipe,$members,$calibration" "$dest/request.csv"
  grep -Fxq "$sha" "$dest/code-revision.txt"
  IFS=, read -r oy oi status reason < "$dest/outcome.csv"
  test "$oy,$oi" = "$y,$id"
  [[ "$status" == measured || "$status" == blocked ]]
  if [[ "$status" == measured ]]; then
    test "$reason" = independent-audit-pass
    gzip -dc "$dest/predictions.csv.gz" > "$root/fit/exports/$key.csv"
    "$root/fit/audit" "$root/fit/cohorts/$y/matched-targets.csv" "$root/fit/exports/$key.csv" "$dest/metrics.txt" "$y" "$id" matched "$root/fit/registry/$key-42.csv"
    cmp "$root/fit/registry/$key-42.csv" "$dest/independent-metrics.csv"
    bash scripts/c015-cost.sh "$y" "$id" "$id" 42 "$dest" > "$root/fit/registry/$key-42-cost.csv"
    if [[ "$id" == P13 ]]; then
      gzip -dc "$dest/distribution.csv.gz" > "$dest/distribution.csv"
      "$root/p13-audit" "$root/fit/joined" "$dest" "$y" | tee "$root/final/$key-audit.txt"
      cmp "$root/final/$key-audit.txt" "$dest/p13-audit.txt"
      awk -F, 'FNR==1{next} FILENAME==ARGV[1]{p[$3 SUBSEP $4]=$6;t[$3 SUBSEP $4]=$5;next} {k=$1 SUBSEP $2;if(!(k in p)||p[k]+0!=$4+0||t[k]+0!=$3+0)exit 1;delete p[k]} END{if(length(p))exit 1}' "$root/fit/exports/$key.csv" "$dest/distribution.csv"
      rm "$dest/distribution.csv"
      cp "$dest/probabilistic-metrics.csv" "$root/final/$key-probabilistic.csv"
    fi
    cp "$dest/parameters.csv" "$root/final/$key-parameters.csv"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$y" "$id" "$status" "$reason" "$combrun" "$jid" "$aid" "$sha" >> "$root/final/combinations.csv"
done < campaigns/015-tournament/combination-matrix.csv
test "$(wc -l < "$root/final/combinations.csv")" -eq 17
"$root/rank" "$root/fit/registry" "$root/final"
"$root/bootstrap" "$root/fit/exports" > "$root/final/bootstrap.csv"
cp "$root/fit/registry/fits.csv" "$root/final/fits.csv"
# Archive annual independent summaries and original evidence digests, separately
# from large transient prediction artifacts.
mkdir -p "$root/final/annual-evidence"
cp "$root/fit/registry/"*.csv "$root/final/annual-evidence/"
printf 'fit_run,%s\ncombination_run,%s\n' "$fitrun" "$combrun" > "$root/final/provenance.csv"
for dir in "$root/fit/source" "$root/comb"; do
  find "$dir" -mindepth 2 -maxdepth 2 -name SHA256SUMS -print0 | sort -z | while IFS= read -r -d '' f; do
    key=$(basename "$(dirname "$f")");cp "$f" "$root/final/annual-evidence/$key.sha256"
  done
done
(cd "$root/final";find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS;sha256sum -c SHA256SUMS)
cat "$root/final/ranking.csv"
cat "$root/final/bootstrap.csv"
echo "c015-finalize,$fitrun,$combrun,pass"
