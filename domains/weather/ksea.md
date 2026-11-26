# KSEA dataset manifest

- Station: Seattle–Tacoma International Airport, IEM SEA / WA_ASOS.
- Coordinates: 47.4447, -122.3144; elevation 137 m.
- Station metadata: https://mesonet.agron.iastate.edu/sites/site.php?network=WA_ASOS&station=SEA
- Source: https://mesonet.agron.iastate.edu/request/download.phtml?network=WA_ASOS
- API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=
- Raw routine reports (report_type=3), UTC, annual CSV requests 2000–2025; start inclusive, end exclusive.
- Variables: tmpf (F), dwpf (F), relh (%), drct (degrees true), sknt (knots), alti (inHg), mslp (millibar). Missing code M.
- IEM historical archive is minimally quality controlled. Audit range, duplicate/conflicting timestamps and coverage; never silently invent values.
- Download script: scripts/fetch-ksea.sh; caches one annual file and SHA-256 sidecar per year. Existing verified files are reused, not re-downloaded.
- Snapshot: acquired 2000–2025; [26 frozen checksums](../../results/AEV-009/snapshot.sha256), [coverage and scores](../../results/AEV-009/README.md). Data rights/attribution review pending before redistributing raw bytes.

## Origin-time covariate audit

The first weather feature comparison will use the same selected routine report as the origin temperature: dew point converted F to C, altimeter setting in hPa (not sea-level pressure), and horizontal wind vector in m/s (meteorological direction denotes where wind comes from). Calm speed zero implies zero vector even if direction missing. Missing or malformed/out-of-broad-range non-temperature values are kept missing, counted rather than imputed. Audit joint coverage on the 72+24 complete temperature cohort, and require identical temperature timestamps/values to the original parser. No future covariate availability is required. No model configuration chosen here; calendar results and joint coverage must be checked first.
