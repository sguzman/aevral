#!/usr/bin/env bash
# Extract source-model timing, not CI scheduling/setup wall time.
# Original timers differ in whether they include export; retain semantics.
set -euo pipefail
year="${1:?year}"
id="${2:?id}"
recipe="${3:?recipe}"
seed="${4:?seed}"
dir="${5:?artifact directory}"
rss=$(awk -F: '/Maximum resident set size \(kbytes\):/{gsub(/^[[:space:]]+/,"",$2);print $2;n++}END{if(n!=1)exit 1}' "$dir/resources.txt")
[[ "$rss" =~ ^[0-9]+$ ]]
awk -F, -v y="$year" -v id="$id" -v r="$recipe" -v seed="$seed" -v rss="$rss" '
  $2==y&&$3==r{
    if(id=="I"&&$1=="fit-seconds"){fit=$4;nf++}
    if(id=="I"&&$1=="score"){pred=$7;np++}
    if(id=="G"&&$1=="upstream-config"&&$4==seed){fit=$NF;nf++;pred=0;np++}
    if(id=="K"&&$1=="gp-config"){fit=$(NF-1);pred=$NF;nf++;np++}
    if(id=="N"&&$1=="strategy-fit"&&$4==seed){fit=$(NF-1);pred=$NF;nf++;np++}
    if(id=="W"&&$1=="representation-fit"&&$4==seed){fit=$(NF-1);pred=$NF;nf++;np++}
    if(id~/^[LRMSAHX]$/&&$1=="optimization-cost"){fit=$4;pred=$5;nf++;np++}
    if(id~/^(E08|E13|E14|P13)$/&&$1=="combination-cost"){fit=$4;pred=$5;nf++;np++}
  }
  END{
    if(nf!=1||np!=1||fit!~/^[0-9]+([.][0-9]+)?$/||pred!~/^[0-9]+([.][0-9]+)?$/)exit 1
    printf "%.9f,%s\n",fit+pred,rss
  }' "$dir/metrics.txt"
