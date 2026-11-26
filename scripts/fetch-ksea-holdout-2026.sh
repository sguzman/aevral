#!/usr/bin/env bash
# AEV-015 only: immutable partial-year archive, cutoff predeclared in FROZEN.md.
set -euo pipefail
root="${AEVRAL_DATA_DIR:-data/raw/ksea}"
mkdir -p "$root"
cd "$root"
file=SEA-2026.csv
if [[ -e "$file" || -e "$file.sha256" ]]; then
  [[ -s "$file" && -s "$file.sha256" ]] && sha256sum -c "$file.sha256" || {
    echo "Incomplete or corrupted holdout cache; refusing replacement" >&2
    exit 1
  }
  exit 0
fi
temp="$(mktemp "./$file.XXXXXX")"
trap 'rm -f "$temp"' EXIT
url="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=WA_ASOS&station=SEA&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&year1=2026&month1=1&day1=1&year2=2026&month2=11&day2=1"
curl --fail --location --retry 3 --retry-delay 3 --max-time 300 --output "$temp" "$url"
grep -Eq '^station,valid,.*tmpf|^station,valid,tmpf' "$temp" && grep -q '^SEA,' "$temp" || { echo "Unexpected holdout response" >&2; exit 1; }
mv "$temp" "$file"
sha256sum "$file" > "$file.sha256"
printf '%s\n' "$url" > "$file.url"
date -u +%FT%TZ > "$file.retrieved"
