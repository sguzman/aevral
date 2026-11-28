# C018 — Bounded LAX deployment protocol
Frozen before LAX historical assessment. No model tuning or candidate search.
## Data
IEM CA_ASOS LAX (ICAO KLAX), routine report_type=3, UTC, identical weather fields to KSEA. Separate annual snapshots 2000–2025, each raw file with SHA256, URL and retrieval timestamp; no assessment-derived selection. KSEA uses existing frozen C015 snapshot and legacy conflict policy. LAX starts with strict conflict handling; if duplicates conflict, audit counts and quarantine exact conflicting timestamps by a declared station-specific policy, never choose a report by outcome. Missing observations remain missing.
## Fixed evaluation
- Assessment: 2025-01-01 00:00 UTC to 2026-01-01 00:00 UTC, C015 strict horizon boundary (origin+60 >= start; origin+1440 < end).
- Training: complete labels strictly before 2025-01-01; all available valid training origins from 2000 onward. Seattle transfer uses Seattle-fitted transforms on LAX inputs. No Seattle refit on LAX.
- LAX P13 calibration: 2022, 2023, 2024 annual out-of-fold predictions; each year's four members trained only on labels strictly before its start. Final members fitted before 2025.
- Fixed seed42, I/L/R/M and P13 recipe unchanged. No other candidates or hyperparameter search.
- Compare daily persistence, KSEA incumbent transfer, LAX incumbent, LAX P13 on identical complete LAX origins. Record possible, matched, missing and quarantined coverage, per-lead and pooled MAE/RMSE Celsius; P13 CRPS and 50/80/90/95% coverage and widths. Report point estimates, no claims of independent prospective skill.
## Live forecast
Only after assessment is archived: extend LAX snapshot through the latest available routine observations; train final bundle with the same three calibration years (2022–24) and strict cutoff at the chosen completed UTC issue hour. No use of labels at or beyond cutoff. Forecast next 24 hours only from exact complete causal origin; if current hour lacks needed observations, report unavailability, not a silently backdated live forecast. Archive CSV, issue, model and source hashes, retrieval UTC, and CI provenance. A historical backdated example must be explicitly labeled.
## Acceptance
Verified CLI file roundtrip, real frozen model prediction parity (no new tuning), bounded matched LAX evaluation, saved artifact reload and forecast or explicitly documented current data limitation. Stop after acceptance; no economics campaign until separately approved.
