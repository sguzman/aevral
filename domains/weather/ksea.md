# KSEA dataset manifest (planned)

- Station: Seattle–Tacoma International Airport, IEM SEA / WA_ASOS.
- Coordinates: 47.4447, -122.3144; elevation 137 m.
- Station metadata: https://mesonet.agron.iastate.edu/sites/site.php?network=WA_ASOS&station=SEA
- Source: https://mesonet.agron.iastate.edu/request/download.phtml?network=WA_ASOS
- API: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=
- Raw routine reports (report_type=3), UTC, annual CSV requests 2000–2025; start inclusive, end exclusive.
- Variables: tmpf (F), dwpf (F), relh (%), drct (degrees true), sknt (knots), alti (inHg), mslp (millibar). Missing code M.
- IEM historical archive is minimally quality controlled. Audit range, duplicate/conflicting timestamps and coverage; never silently invent values.
- Download script: scripts/fetch-ksea.sh; caches one annual file and SHA-256 sidecar per year. Existing verified files are reused, not re-downloaded.
- Snapshot status: not acquired; hashes and actual coverage pending. Data rights/attribution review pending before redistributing raw bytes.
