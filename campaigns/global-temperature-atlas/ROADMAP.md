# Global temperature predictability atlas — deferred proposal

Status: **parked, planning only** (2026-11-27). Campaign number and execution order intentionally unassigned. User approved writing this roadmap, not acquiring observations, training models, launching CI, or changing the current C016–C018 scope. Start only after C018 closure and separate user approval; may precede or follow macroeconomic research.

## Research questions

1. Which observation stations have the lowest attainable 1–24-hour temperature forecast errors under the same frozen Aevral models and data contract?
2. Where does machine learning improve most relative to daily persistence, climatology and current-temperature persistence?
3. Are differences associated with climate regime, ocean influence, daily and annual temperature amplitude, cloud/rain variability, frontal passages, station altitude or data completeness?
4. Does P13 outperform its local MLP incumbent across distinct climate regimes, and are its intervals calibrated?

Motivation: C018's preliminary locally trained LAX MLP MAE is 0.945110265288°C on 177,744 targets; daily persistence 1.225245921727°C, Seattle transfer 1.810404327173°C. These are producer results pending independent audit at proposal time; reference the eventual C018 report, not this preliminary note, for final values. No city in the candidate list has a measured result from this proposal.

## Candidate universe (169 places)

Groups are climate hypotheses and sampling strata, not established rankings or verified station metadata. A city is a geographic label; an actual forecast is for one named observation site, often an airport outside the city. Some labels are nearby or share an airport catchment (e.g. Lima/Callao); resolve and deduplicate physical stations before any fitting. Where no suitable station exists, record the exclusion. Source availability, geography, exact regime, identifiers, observation cadence and station moves remain unverified individually.

### A — coastal deserts and subtropical marine moderation
Hypothesis: persistent marine boundary layers, low precipitation or mild synoptic variability may produce small absolute errors; cloud transitions and sea breezes can break that pattern.

- Lima, Peru
- Callao, Peru
- Trujillo, Peru
- Chiclayo, Peru
- Arica, Chile
- Iquique, Chile
- Antofagasta, Chile
- La Serena, Chile
- San Diego, USA
- Santa Barbara, USA
- Oxnard, USA
- Ensenada, Mexico
- Tijuana, Mexico
- Agadir, Morocco
- Essaouira, Morocco
- Casablanca, Morocco
- Nouadhibou, Mauritania
- Dakar, Senegal
- Walvis Bay, Namibia
- Lüderitz, Namibia
- Swakopmund, Namibia
- Muscat, Oman
- Salalah, Oman

### B — tropical and subtropical islands
Hypothesis: ocean moderation may reduce temperature extremes; convection, trade wind showers and terrain can introduce errors.

- Honolulu, USA
- Hilo, USA
- Kahului, USA
- Lihue, USA
- San Juan, Puerto Rico
- Bridgetown, Barbados
- Oranjestad, Aruba
- Willemstad, Curaçao
- George Town, Cayman Islands
- Nassau, Bahamas
- Ponta Delgada, Portugal
- Funchal, Portugal
- Las Palmas de Gran Canaria, Spain
- Santa Cruz de Tenerife, Spain
- Praia, Cabo Verde
- Mindelo, Cabo Verde
- Saint-Denis, Réunion, France
- Port Louis, Mauritius
- Victoria, Seychelles
- Malé, Maldives
- Nadi, Fiji
- Apia, Samoa
- Papeete, French Polynesia, France
- Nouméa, New Caledonia, France
- Hagåtña, Guam, USA
- Koror, Palau
- Majuro, Marshall Islands
- Rarotonga, Cook Islands

### C — equatorial coastal and tropical urban climates
Hypothesis: low annual temperature variation, but rainfall timing and cloud variability may outweigh that advantage.

- Singapore
- Kuala Lumpur, Malaysia
- Penang, Malaysia
- Kuching, Malaysia
- Bandar Seri Begawan, Brunei
- Kota Kinabalu, Malaysia
- Jakarta, Indonesia
- Denpasar, Indonesia
- Surabaya, Indonesia
- Makassar, Indonesia
- Manila, Philippines
- Cebu, Philippines
- Davao, Philippines
- Ho Chi Minh City, Vietnam
- Colombo, Sri Lanka
- Mombasa, Kenya
- Dar es Salaam, Tanzania
- Accra, Ghana
- Abidjan, Côte d’Ivoire
- Libreville, Gabon
- Douala, Cameroon
- Belém, Brazil
- Fortaleza, Brazil
- Natal, Brazil
- Recife, Brazil
- Cartagena, Colombia
- Barranquilla, Colombia
- Panama City, Panama
- Guayaquil, Ecuador
- Georgetown, Guyana
- Cairns, Australia
- Darwin, Australia

### D — clear-sky deserts and seasonally dry interiors
Hypothesis: repeatable diurnal heating may yield high baseline or model skill, despite larger temperature ranges; fronts, dust, monsoons and heat waves are challenges.

- Phoenix, USA
- Tucson, USA
- Yuma, USA
- Las Vegas, USA
- Palm Springs, USA
- Mexicali, Mexico
- Hermosillo, Mexico
- La Paz, Mexico
- Dubai, UAE
- Abu Dhabi, UAE
- Doha, Qatar
- Manama, Bahrain
- Kuwait City, Kuwait
- Riyadh, Saudi Arabia
- Jeddah, Saudi Arabia
- Aqaba, Jordan
- Eilat, Israel
- Hurghada, Egypt
- Aswan, Egypt
- Luxor, Egypt
- Alice Springs, Australia
- Port Hedland, Australia
- Broome, Australia

### E — tropical highlands and plateaus
Hypothesis: small seasonal swings and repeated day/night heating can be favorable; local convection and altitude effects may dominate.

- Mexico City, Mexico
- Guadalajara, Mexico
- Puebla, Mexico
- Cuernavaca, Mexico
- San José, Costa Rica
- Guatemala City, Guatemala
- Medellín, Colombia
- Bogotá, Colombia
- Quito, Ecuador
- Cuenca, Ecuador
- Arequipa, Peru
- Nairobi, Kenya
- Addis Ababa, Ethiopia
- Kigali, Rwanda
- Kampala, Uganda
- Arusha, Tanzania
- Bangalore, India
- Bandung, Indonesia

### F — mild maritime and Mediterranean climates
Hypothesis: oceanic moderation may lower absolute errors; storm tracks, fronts and local wind make seasonal performance important.

- San Francisco, USA
- Monterey, USA
- Lisbon, Portugal
- Faro, Portugal
- Málaga, Spain
- Alicante, Spain
- Valencia, Spain
- Palma, Spain
- Nice, France
- Palermo, Italy
- Cagliari, Italy
- Valletta, Malta
- Larnaca, Cyprus
- Cape Town, South Africa
- Perth, Australia
- Adelaide, Australia
- Sydney, Australia
- Auckland, New Zealand
- Wellington, New Zealand
- Hobart, Australia
- Valparaíso, Chile
- Montevideo, Uruguay
- Mar del Plata, Argentina

### G — anchors and challenging controls
Use these to avoid a low-variance-only sample and contextualize conclusions. KSEA and KLAX are already exposed anchor stations; do not call their existing evaluations fresh holdouts.

- Seattle, USA
- Los Angeles, USA
- Chicago, USA
- Minneapolis, USA
- Denver, USA
- Miami, USA
- New York, USA
- Winnipeg, Canada
- Reykjavík, Iceland
- Dublin, Ireland
- London, UK
- Helsinki, Finland
- Tokyo, Japan
- Seoul, South Korea
- Ulaanbaatar, Mongolia
- New Delhi, India
- Bangkok, Thailand
- Dhaka, Bangladesh
- Johannesburg, South Africa
- Melbourne, Australia
- Buenos Aires, Argentina
- Santiago, Chile

## Proposed execution gates (not authorization)

| Gate | Scope and deliverable |
|---|---|
| G0 — preregistration | Freeze target, station eligibility, source, temporal splits, sampling rule, resource budget, metrics and maximum number of fits. Choose ordering relative to macro research. |
| G1 — catalog | Verify station identity/ICAO/source ID, lat/lon/elevation, station relocation and instrument history, time zone, hourly cadence, UTC encoding, units, licensing and coverage. Dedupe and record all inclusion/exclusion decisions without using assessment errors. |
| G2 — source adapter | Prefer existing IEM routine METAR path; international availability varies. If needed, build a Rust NOAA ISD/global-hourly adapter and verify temperature/dew point, wind vector and comparable altimeter pressure semantics (do not substitute sea-level/station pressure silently). Compare overlapping KSEA/KLAX source outputs, timestamp and quantization differences before mixing sources. |
| G3 — bounded pilot | Propose 12 distinct stations: San Diego, Honolulu, Lima/Callao (one station), Arica, Funchal, Singapore, Malé, Mombasa, Phoenix, Quito, Chicago, Auckland. Finalize on coverage only; KLAX/KSEA are exposed anchors. No pilot until approved. |
| G4 — breadth-first screen | At most one fixed locally trained incumbent per eligible station, plus baselines. Proposed initial 12-station budget; expand in separately approved batches (e.g. 12–24 stations) up to the eligible catalog. Publish failures and excluded stations. |
| G5 — frozen P13 follow-up | Predefine a stratified subset using development-only results and coverage (include low-error, high-skill, diverse climates and controls). Fit 4 members and prior-only calibration; no new hyperparameter search. Freeze selection before seeing final confirmation outcomes. |
| G6 — confirmation | Re-evaluate selected stations on untouched time periods or prospective ledger; report winner's curse and multiplicity. No claim of globally lowest error outside the sampled stations. |
| G7 — atlas | Station catalog, open metrics and metadata, per-lead curves, absolute and relative-skill maps, uncertainty coverage, seasonal profiles, reproducibility guide and report. |

## Proposed common benchmark

- Target: hourly near-surface air temperature (°C); direct 1–24h outputs from exact completed UTC origins; 72-hour causal history and identical 80-feature representation for incumbent and P13. Keep all model recipes, seeds and budgets frozen. Local training-only scaling; same model architecture does not mean identical fitted weights.
- Proposed common chronological reference: training 2000–2024 with annual 2022–24 calibration for P13; assessment calendar 2025. Dates are provisional until common-source coverage is audited. Report common-history sensitivity (same start/cutoff and training budget); never select a shorter history by assessment error. For genuinely untouched confirmation freeze a later period independent of screening; existing KSEA and KLAX are exposed.
- Assess each station on its own complete cohort; also compare synchronized common-origin subsets by station group where feasible and local-season/hour strata. Do not discard difficult hours or stations on observed model errors. Set minimum coverage rules before fitting; report denominator and exclusions, including missingness by season and hour. Do not silently fill target gaps.
- Primary absolute ranking: pooled MAE across all 24 leads; secondary RMSE, median absolute error, 90th/95th percentile absolute error, lead-24 MAE and seasonal/lead MAE. Equal station weights in climate-group summaries; no pooled cross-station ranking biased by more observed hours.
- Relative ranking: 1 − MAE(model)/MAE(daily persistence); also compare current-temperature persistence and training-only local hour/day-of-year climatology. If baseline MAE is zero, skill undefined; report absolute errors instead. Report training-only temperature scale, daily amplitude, anomaly MAE, and normalized errors with explicit denominator. Small absolute MAE does not imply high skill.
- P13: CRPS, empirical 50/80/90/95% coverage and mean width; compare calibrated intervals across stations, not just median error. Block-bootstrap uncertainty by origin day (1/7/14-day sensitivity), preserving all leads per origin; distinguish descriptive intervals from confirmatory claims.
- Capture measurement resolution, timestamp/report-type bias, coastal airport siting, heat islands, extreme events, missingness selection, seasonal shifts and data provider discrepancies. Keep forecast model inputs observational (no future reanalysis or numerical weather forecasts).
- Record every source SHA256, URL, retrieval timestamp, station metadata, transform and artifact version; publish finite per-station runtime, fit count, coverage, errors, seeds and outcome. Persist tables and hashes in Git, large artifacts in designated external durable storage/cache with retention stated.

## Data source references

- Iowa Environmental Mesonet global airport METAR archive (routine report option): https://www.mesonet.agron.iastate.edu/request/download.phtml?network=AI__ASOS
- NOAA NCEI Integrated Surface Database / Global Hourly: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
- NOAA Global Hourly access: https://www.ncei.noaa.gov/access/search/data-search/global-hourly/

These sources are plausible global observation archives, not proof that any particular station meets the contract. No station data downloaded for this proposal.
