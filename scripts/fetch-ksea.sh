#!/usr/bin/env bash
# Fetch immutable-by-default annual cache entries. Run with bash.
set -euo pipefail
root="${AEVRAL_DATA_DIR:-data/raw/ksea}"
first="${1:-2000}"
last="${2:-2025}"
[[ "$first" =~ ^[0-9]{4}$ && "$last" =~ ^[0-9]{4}$ ]] || exit 2
(( first >= 2000 && last <= 2025 && first <= last )) || exit 2
mkdir -p "$root"
cd "$root"
for ((year=first; year<=last; year++)); do
  file="SEA-$year.csv"
  if [[ -e "$file" || -e "$file.sha256" ]]; then
    [[ -s "$file" && -s "$file.sha256" ]] && sha256sum -c "$file.sha256" || {
      echo "Incomplete or corrupted cache: $file; refusing automatic replacement" >&2
      exit 1
    }
    continue
  fi
  temp="$(mktemp "./$file.XXXXXX")"
  trap 'rm -f "$temp"' EXIT
  url="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=WA_ASOS&station=SEA&data=tmpf,dwpf,relh,drct,sknt,alti,mslp&tz=Etc/UTC&format=onlycomma&missing=M&report_type=3&sts=$year-01-01T00:00:00Z&ets=$((year+1))-01-01T00:00:00Z"
  curl --fail --location --retry 3 --retry-delay 3 --max-time 300 --output "$temp" "$url"
  grep -q '^station,valid,.*tmpf' "$temp" && grep -q '^SEA,' "$temp" || {
    echo "Unexpected or empty response for $year" >&2; exit 1;
  }
  mv "$temp" "$file"
  sha256sum "$file" > "$file.sha256"
  printf '%s\n' "$url" > "$file.url"
  date -u +%FT%TZ > "$file.retrieved"
  trap - EXIT
  sleep 2
done
