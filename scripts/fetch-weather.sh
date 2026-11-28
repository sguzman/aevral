#!/usr/bin/env bash
# IEM routine observations, immutable per requested UTC date interval.
set -euo pipefail
if [[ $# != 4 && $# != 5 ]]; then
  echo 'Usage: fetch-weather.sh KLAX|KSEA YYYY-MM-DD YYYY-MM-DD OUTPUT_DIR [YYYY-MM-DDTHH:00 cutoff UTC] (exclusive end)' >&2
  exit 2
fi
case "$1" in
  KLAX) station=LAX; network=CA_ASOS;;
  KSEA) station=SEA; network=WA_ASOS;;
  *) echo 'Unregistered ICAO station' >&2; exit 2;;
esac
start="$2"; end="$3"; root="$4"; cutoff="${5:-}"
for d in "$start" "$end"; do
  [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ && "$(date -u -d "$d" +%F)" == "$d" ]] || exit 2
done
[[ "$start" < "$end" && "$start" > "1999-12-31" ]] || exit 2
if [[ -n "$cutoff" ]]; then
  [[ "$cutoff" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:00$ ]] || exit 2
  [[ "$(date -u -d "$cutoff" +%FT%H:%M)" == "$cutoff" && "$cutoff" < "$(date -u +%FT%H:01)" ]] || exit 2
  [[ ! "$start" > "${cutoff:0:10}" && "$end" == "$(date -u -d "${cutoff:0:10} +1 day" +%F)" ]] || exit 2
else
  [[ "$end" < "$(date -u -d tomorrow +%F)" ]] || exit 2
fi
# Prevent large accidental requests; historical acquisition can use annual slices.
(( ($(date -u -d "$end" +%s)-$(date -u -d "$start" +%s))/86400 <= 366 )) || exit 2
mkdir -p "$root"
cd "$root"
file="$station-$start-$end.csv"
if [[ -n "$cutoff" ]]; then file="$station-$start-through-${cutoff//:/}.csv"; fi
url="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=$network&station=$station&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&year1=${start:0:4}&month1=${start:5:2}&day1=${start:8:2}&year2=${end:0:4}&month2=${end:5:2}&day2=${end:8:2}"
validate() {
  awk -F, -v s="$station" -v a="$start" -v b="$end" -v cutoff="${cutoff/T/ }" '
    /^#/ || /^$/ {next}
    !header {
      for(i=1;i<=NF;i++){gsub(/\r$/,"",$i); if($i in cols)exit 1; cols[$i]=i}
      if(!("station" in cols)||!("valid" in cols)||!("tmpf" in cols)||!("dwpf" in cols)||!("alti" in cols)||!("drct" in cols)||!("sknt" in cols))exit 1
      header=NF;next
    }
    NF!=header || $(cols["station"])!=s {exit 1}
    {d=substr($(cols["valid"]),1,10);if(d<a||d>=b||(cutoff!=""&&$(cols["valid"])>cutoff))exit 1;n++}
    END{if(!header||!n)exit 1}
  ' "$1"
}
if [[ -e "$file" || -e "$file.sha256" || -e "$file.url" || -e "$file.retrieved" ]]; then
  [[ -s "$file" && -s "$file.sha256" && -s "$file.url" && -s "$file.retrieved" ]] || { echo 'Incomplete cache; refusing replacement' >&2; exit 1; }
  [[ "$(cat "$file.url")" == "$url" ]] || exit 1
  sha256sum -c "$file.sha256"
  validate "$file"
  exit 0
fi
temp="$(mktemp "./$file.XXXXXX")"
trap 'rm -f "$temp" "$temp.filtered"' EXIT
curl --fail --location --retry 2 --retry-delay 3 --max-time 300 --output "$temp" "$url"
if [[ -n "$cutoff" ]]; then
  # The provider may return later reports in the same requested UTC day.
  # Keep only records at or before the explicitly frozen issue instant.
  awk -F, -v cutoff="${cutoff/T/ }" '
    /^#/ || /^$/ {print;next}
    !header {for(i=1;i<=NF;i++)if($i=="valid")t=i;if(!t)exit 1;header=1;print;next}
    $t<=cutoff {print}
    END{if(!header)exit 1}
  ' "$temp" > "$temp.filtered"
  mv "$temp.filtered" "$temp"
fi
validate "$temp" || { echo 'Unexpected station, schema, date interval, or empty response' >&2; exit 1; }
# Do not overwrite a cache file written by another invocation.
ln "$temp" "$file"
sha256sum "$file" > "$file.sha256"
printf '%s\n' "$url" > "$file.url"
date -u +%FT%TZ > "$file.retrieved"
echo "source,$1,$station,$network,$start,$end,$file"
