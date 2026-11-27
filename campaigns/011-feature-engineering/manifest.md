# C011 — Frozen same-archive representation manifest

Frozen before implementation or scoring. 29 recipes: one matched-budget control and two per 14 feature families. All measured screens advance, at most 29 finalists, each four folds and three seeds, reusing the 2019 seed42 screen. Maximum 319 additional annual jobs, 348 total; sequential batches at most 96 jobs, at most six simultaneous fits. No new data, no tuning after screens, no further recipes without approval. C011 is a representation study, not a replacement of the fixed 80-input leaderboard.

## Data and cohorts

Use the exact C003 frozen archive, hashes, hourly report selection and fold partition. 72 consecutive temperature hours including origin, next 24 target hours, and complete origin dew/pressure/wind produce the fixed A cohorts: 2019 121849/7455 train/eval, 2020 129327/7543, 2021 136893/7501, 2022 144417/7639. Load only through selected fold end; no future covariates, no 2023–26 development scores. Origin-time feature API consumes only bins at or before origin; target extraction is a distinct operation. Retain unchanged C003 normalization for the existing 80 columns and targets.

All fits use the identical matched A training origin list, and the identical 8192 chronological evenly spaced subset as C010 (index floor(i*N/8192)). All primary scores use exactly the A evaluation origins. A-derived uses only A origin information; B-expanded uses extra historical observations from the same archive and is reported separately. B-expanded adds optional extras with explicit missingness masks, never silently removes a matched origin. Each optional scalar is standardized using observed values among the selected training rows; missing becomes zero in standardized units, followed by binary availability (1 present, 0 absent). No future interpolation. Report observed/missing counts per optional column and all-optional-observed coverage separately for train/eval; a mask is not a claim the data existed. Natural coverage is evaluated using the same fitted model, no refit: for suffix24/48, require only its visible consecutive temperature history, 24 target hours and complete origin weather; report full natural evaluation metrics/counts as well as matched metrics. All other recipes retain the original A eligibility (including feature-selection recipes, by design); their natural cohort equals A. Optional histories do not require complete intervening hours. Report natural possible/count and matched overlap, plus full-history availability for B. No comparing unmatched scores as though paired. No new station, data, external forecast, or extra meteorological columns.

## Model and training protocol

Native Rust existing arbitrary-width Dense tanh MLP, dimensions input→64→24, Xavier initialization, direct normalized MSE and Adam 0.001, 12 epochs, same splitmix shuffle and seeds 42,123,2026 as C010 F01 joint-mlp64. One update per row, 98304 per full fit. Only representation changes; varying input width changes parameter count/compute, disclose. C010 direct MLP64 is exact matched-budget control and must reproduce per-fold/seed scores (2019 seed42 MAE 1.52449139, RMSE 1.98742079; tolerance 1e-7 °C); retain C003 full-budget incumbent and C010 MAE N-HiTS as separately labeled references, not controlled comparisons. No requirement to beat incumbent.

Original features in order: normalized temperatures oldest→newest (72), original daily sin/cos, annual sin/cos (4), standardized dew point, altimeter-derived pressure, east/north wind (4). Lag 0 is current observed temperature, lag h means exact bin h hours earlier. Unless explicitly replacing/selecting, append features after original 80. Fit engineered nonbinary continuous column mean and population SD on selected matched training rows only, using observed entries only; zero variance or no observations gives scale 1 and mean observed mean or 0, flag in logs. Binary masks/indicators remain 0/1. Original columns unchanged. All column orders are family recipe order below; within a list use listed order. Optional scalars interleave value, availability. Natural suffix input uses original fold scales, not separately fitted scales.

## Frozen screen matrix

| Method | Track | Recipe | Inputs | Representation |
|---|---|---|---:|---|
| R00 | A-control | base80 | 80 | unchanged C003 input |
| R01 | A-derived | suffix24 | 32 | last 24 temperatures plus original calendar and origin weather |
| R01 | A-derived | suffix48 | 56 | last 48 temperatures plus original calendar and origin weather |
| R02 | B-expanded | sparse-lags-14d | 92 | append lag hours 72,95,119,143,167,335 |
| R02 | B-expanded | sparse-lags-28d | 96 | append lag hours 72,95,119,143,167,335,503,671 |
| R03 | A-derived | rolling-6-24-72 | 89 | append mean, range, population variance for 6,24,72 observed temperatures |
| R03 | A-derived | rolling-3-12-48 | 89 | append mean, range, population variance for 3,12,48 observed temperatures |
| R04 | A-derived | gradient-short | 86 | append slope and acceleration for lag intervals 1,3,6 |
| R04 | A-derived | gradient-long | 86 | append slope and acceleration for lag intervals 6,12,24 |
| R05 | A-derived | harmonics-2-3 | 88 | append daily and annual sine/cosine at harmonics 2,3 |
| R05 | A-derived | harmonics-2-4 | 92 | append daily and annual sine/cosine at harmonics 2,3,4 |
| R06 | A-derived | dew-depression-add | 81 | append current T minus dew point |
| R06 | A-derived | dew-depression-replace | 80 | replace current dew-point column with current T minus dew point |
| R07 | B-expanded | pressure-tendency-short | 86 | append current pressure minus pressure at lags 3,6,12 hours divided by lag hours |
| R07 | B-expanded | pressure-tendency-long | 86 | same for 6,24,48 hours |
| R08 | A-derived | wind-speed | 81 | append current wind-vector magnitude |
| R08 | A-derived | wind-unit | 84 | append speed, two unit-vector components, calm indicator |
| R09 | B-expanded | weather-history-short | 104 | append dew point, pressure, wind east/north at lags 1,3,6 hours |
| R09 | B-expanded | weather-history-long | 104 | same at lags 6,12,24 hours |
| R10 | A-derived | anomaly-month-hour | 104 | append anomalies for last 24 temperatures relative to training climatology keyed by UTC month/hour |
| R10 | A-derived | anomaly-month | 104 | append anomalies for last 24 temperatures relative to training climatology keyed by UTC month |
| R11 | A-derived | spectral-3 | 86 | append centered complex Fourier coefficients for periods 24,12,8 hours |
| R11 | A-derived | spectral-5 | 90 | same for periods 24,12,8,6,4 hours |
| R12 | A-derived | interactions-local | 86 | append six specified current/lag/weather interactions |
| R12 | A-derived | interactions-calendar | 92 | append twelve specified calendar/current/weather interactions |
| R13 | A-derived | select32-mean | 32 | top 32 base features by absolute Pearson correlation with mean of 24 training targets |
| R13 | A-derived | select48-leads | 48 | top 48 base features by average absolute Pearson correlation across 24 training targets |
| R14 | B-expanded | history96 | 128 | append lags 72 through 95 inclusive, each with availability mask |
| R14 | B-expanded | history168 | 272 | append lags 72 through 167 inclusive, each with availability mask |

### Exact definitions
- R01: slice original temperatures suffix and retain original 8 non-temperature columns. No additional missing flags.
- R02/R14: optional raw Celsius lag at exact hourly bin; no fallback to neighboring report. R14 order increasing lag (newer to older). Natural eligibility still base A.
- R03: for each listed window, use raw Celsius population variance (denominator n), mean and max-minus-min; output order mean, range, variance.
- R04: for each interval h, slope (T0−Th)/h and acceleration (T0−2Th+T2h)/h², both causal, raw Celsius.
- R05: recover daily/annual phase from original calendar convention (UTC minute, actual Gregorian year length), append sin/cos of k times phase, daily then annual for each k.
- R06: raw Celsius depression, replace only standardized dew column in replacement recipe; fit depression scale on matched selected training. Do not clamp.
- R07: pressure means existing altimeter-derived hPa, not sea-level pressure assertion. Optional tendency computed only when both exact-bin pressures available, otherwise missing.
- R08: u,v from existing parser, speed sqrt(u²+v²); unit vector u/speed,v/speed when nonzero, zero/zero when calm, calm flag exactly speed=0.
- R09: optional weather values for each listed lag, in dew, pressure, east, north order.
- R10: training-only climatology fit on observed origin temperatures of the full matched training origin list (not targets or evaluation rows), group by UTC calendar month and hour or month. Empty month/hour fallback to training month, then global training-origin mean; empty month fallback global. For each last-24 hourly temperature, oldest→newest, subtract climatology for that historical bin time. No evaluation climatology refit. This is A-derived because origin inputs and deterministic calendar time suffice.
- R11: for each listed period p, let t=0..71 in chronological order, x centered by its 72-hour mean; append (sum x*cos(2πt/p)/72, sum x*sin(2πt/p)/72). No future samples or centered-on-origin filters.
- R12: use original standardized columns, let z0,z1,z24 denote lagged normalized temperatures, d,p,u,v normalized weather, sh,ch,sy,cy original calendar columns. Local pairs ordered (z0,d),(z0,p),(z0,u),(z0,v),(z0,z1),(z0,z24). Calendar pairs: z0 times sh,ch,sy,cy then d times same four then p times same four. Standardize products using training subset.
- R13: selection on the selected 8192 training rows only; absolute Pearson with normalized mean target or average absolute Pearson over each normalized target lead. Constant correlation 0, tie by original index ascending; select k, feed in original column order. Save selected indices and scores; never use evaluation labels. All recipes use same original A cohort for comparison.

## Verification and execution

Implement first R00–R04 (9 screens), then R05–R09 (10), then R10–R14 (10). First gate: independent old/new matched-origin list and target equality, original control exact input/scale and score parity, missing gap handling, suffix natural cohort includes all matched origins, future-temperature/covariate perturbations cannot alter already-issued inputs, training transformations/selection cannot change from evaluation perturbation. Test each feature against hand-calculated synthetic values; explicit calendar/leap boundary, calm wind, missing optional covariates, DFT phase and climatology fallback tests. Require fixed dimensions and finite values, deterministic repeatable initialization and training, no target leakage. CI confirms all 27 hashes, existing legacy regression and tests; export matched per-origin predictions and separate natural predictions where different. Preserve metrics, lead/month metrics, cohort counts and origin-list hashes, scales/selection, parameter count, updates, runtime and peak RSS, code SHA, seed, CI links, artifact metadata. Validate export completeness and scoring; independently rescore raw payloads if available, otherwise disclose. CI artifacts retention 30 days; keep summaries and registry in Git. Model workspace target <=2 GB, job <=30min, <=6 simultaneous, disposable train-only probe (128 rows, 32 predictions), extrapolate fit/inference/feature-preparation cost with 2x safety margin and runtime guard; explicit resource block, no recipe shrinkage. Rank within A-derived/control and B-expanded separately by matched origin-weighted MAE, then pooled RMSE, then recorded fit+predict seconds; report fold/seed variability and natural coverage without naive iid inference. Stop after 348 measured or explicitly resolved outcomes; proceed C012 only after closure.
