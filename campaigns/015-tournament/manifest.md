# C015 — Frozen retrospective execution manifest

Frozen before assessment. Candidate identities and development evidence: [SHORTLIST.md](SHORTLIST.md), commit da299441aa716ed29bbdb9934266b4572a1f3fd1. This manifest specifies the intended interface and gates; the C015 adapters are not yet implemented or tested. No retrospective fit or score has been run.

## Data, chronology and independent cohort gate

Restore exact CI cache key `ksea-routine-utc-2000-2026-jan-oct-v1` at data/raw/ksea, fail on cache miss, verify all 27 files against results/AEV-015/snapshot.sha256 (Git blob 8361c0025b7a98f52916a9d809585cc1e7c35fce). No network refresh or replacement. Load only annual files through the assessment fold; for 2026 use the frozen partial SEA-2026.csv. UTC bounds, inclusive start and exclusive end for all target hours:

| Fold | Start | End |
|---|---|---|
| 2023 | 2023-01-01 00:00 | 2024-01-01 00:00 |
| 2024 | 2024-01-01 00:00 | 2025-01-01 00:00 |
| 2025 | 2025-01-01 00:00 | 2026-01-01 00:00 |
| 2026 | 2026-01-01 00:00 | 2026-11-01 00:00 |

Origin t: train iff t+1440 < start; assessment iff t+60 >= start and t+1440 < end. Discard boundary-straddling targets. Identical C003 72-hour temperature and origin-time meteorology eligibility, scales fitted on complete causal training rows, no future covariate inputs; all 24 observed targets needed for matched assessment. S natural history has its original suffix eligibility; W optional histories use original masks. Preserve C003 first bin convention and dual archive alignment audit.

First implement a separate retrospective constructor and load path, not an unrestricted expansion of Fold::development. Before fitting, a cohort-only CI gate emits ordered train/eval origin files, matched target file (24 per origin), natural S origin list, counts, train-only scales and deterministic subset hashes for 8192/24576/32768 budgets. Independent enumeration from audited hourly bins verifies keys, full target windows, eligibility, target values, masks, scales and subset positions floor(i*N/budget). Commit cohort hashes/counts before fitting; do not display or score retrospective labels in the cohort-only gate. Independently verify actual evaluation keys/targets and actual training subset for every fit, including purged early-stop split. No hardcoded guessed counts. Regression tests preserve all old development guards, known counts and legacy score behavior. Synthetic boundary tests cover leap 2024, December/January, October/November 2026, and held-out input/target perturbation. Missing November/December bins in 2026 are absent (NA), not zero errors.

## Exact execution matrix

Proposed new CLI `aevral retrospective DATA YEAR OUT_CSV ID SEED`; separate `aevral retrospective-cohort DATA YEAR OUT_DIR`. Adapter only reuses/refactors existing recipe implementations; historical commands and validation must continue rejecting retrospective requests. All native fits use cargo --locked --release. G and L use locked Nix official XGBoost 3.0.5 and LightGBM 4.6.0 respectively; verify backend version and existing settings before launch. No Python substitution. Source command suffixes below document exact existing interfaces (not runnable with 2023+ until adapter implemented). DATA=data/raw/ksea, OUT=results-run/predictions.csv.

| ID | Original command after DATA YEAR OUT | Fit cap | Seeds |
|---|---|---:|---|
| I | tournament (no suffix; original mlp64 SGD .005 e30) | full | 42 |
| L | optimization O04-04 final SEED | 32768 | 42 |
| R | optimization O05-02 final SEED | 32768 | 42 |
| M | optimization O01-06 final SEED | 32768 | 42,123,2026 |
| S | optimization O02-06 final SEED | 32768 | 42,123,2026 |
| A | optimization O06-06 final SEED | 32768 | 42 |
| H | optimization O08-06 final SEED | 32768 | 42 |
| X | optimization O07-06 final SEED | 32768 | 42 |
| G | upstream xgboost-128 42 | 24576 | 42 |
| K | gp fitc-s1-m256 | 8192 | 42 |
| N | strategy mae-nhits64 SEED | 8192 | 42,123,2026 |
| W | representation pressure-tendency-short SEED | 8192 | 42,123,2026 |

Matrix = four folds × 20 fits =80 (48 primary, 32 sensitivity). Run original I full-budget recipe and non-fitted daily reference; any persistence/climatology outputs inherited from C003 are descriptive references only and not extra model candidates. No full-budget upgrades for budgeted recipes. No optional fixed-original-train comparison: omitted to preserve finite compute and avoid extra fits. Frozen C012 grid and original model definitions authoritative for hyperparameters, internal split, transforms, seed/shuffle and resource guards. Use original printed recipe labels in exported source files and explicit alias mapping in audit.

Execution audit found hardcoded 2019–2022 gates in tournament.rs, optimization.rs, upstream.rs, gp.rs, neural/strategy.rs, representation.rs, and old shell scripts. C003 score month-edge construction ends at 2022; C013 ensemble month/load/calibration and C014 runner/audit are development-only. Preserve old behavior and expose separate retrospective wrapper(s) over shared numerical code. Do not replace old checks with self-reported counts. Reuse validated cohort registry in retrospective mode. Inspect any additional calendar/guard assumptions during implementation. No claim that these commands work yet.

## Ensemble protocol

Use exact historical C013 joined artifact 15239944097, six SHA256 in results/C013/source-audit.txt. Validate metadata, ZIP and all six hashes, exact historic rows, and previous retrospective joined inputs against durable cohort/prediction hashes. For each year sequentially join seed42 I,L,R,M,S,A,H,X in original order, plus daily and true target, with strict alignment. Fit E13/E14/P13 before loading current assessment targets; history strictly 2020 through year-1 with complete out-of-fold predictions only. Reuse exact C013 E08/E13/E14 and C014 P13 numerical algorithms, no weight or distribution changes. C015 loader validates variable counts, Gregorian leap rules and truncated 2026; reuse original C013/C014 loader for development years where practical. E08 static needs I,L,R,M only; E13/E14 need all eight and complete historical inputs; P13 requires the original four plus complete historical residual calibration. A missing required prediction blocks only dependent outcomes; no substitution or cohort shrinking. Store weights, fit counts, convergence, P13 splits/leaves and support reconstruction evidence. Four combinations × four folds =16 outcomes. P13 issued median is point prediction; exact issued-discrete CRPS, nine pinball quantiles and four central intervals, conditional calibration diagnostics; separate probabilistic results, no fabricated distribution scores for point-only models.

## Ranking, independent audit and uncertainty

Primary: pooled matched Celsius MAE weighted by target count over all four folds, RMSE from pooled squared errors, then fit+predict compute as tie breaker; deterministic exact tie by ID. Track W separately (same matched origins, different information). Sensitivity seed results separate from seed42 ranking; report three-seed MAE range/SD and incomplete seeds, no seed shopping. No complete ranking for partial outcomes. Report all models and families measured historically with source cohort/seed/budget/track labels and negative/blocked statuses; never combine one-year screens or different period averages into a single misleading leaderboard.

Independent Rust or AWK scorer must parse raw exports, enforce ordered exact origin/lead/target agreement with frozen cohort, finite predictions, count, model/fold/seed identity and recompute all overall, 24 lead, observed month and DJF/MAM/JJA/SON errors within 1e-7 Celsius of rounded producer metrics. Season assigned by first predicted UTC hour (same convention as original month diagnostics). No empty month error. P13 independently reconstruct issued supports from parameters and history and recompute proper scores; reconstruction is not independent estimator refitting. Archive full metrics, registry, configuration, code/runner/backend/seed, fit/predict timing, process RSS, coverage, coefficients, hashes, status and artifact IDs. Compare compute/accuracy Pareto front using recorded median annual fit+predict time for complete seed42 outcomes; distinguish shared base-model costs from incremental ensemble costs, no controlled hardware runtime claim.

Paired block bootstrap: per candidate versus I, construct per-origin mean absolute error difference across all 24 leads. Partition each fold separately into non-overlapping fixed UTC calendar blocks anchored at fold start, primary length 7 days; retain nonempty blocks, keep every eligible origin in block together. Within each fold sample the original number of nonempty blocks with replacement, independently across folds but same draws for every matched model; pool by total sampled origin count, not equal block means. 10,000 replicates, SplitMix64 seed 150042, deterministic draw order fold then block; unbiased bounded integer rejection, common draws across candidates. Report point difference and percentile 2.5/97.5 intervals (linear interpolation at index p*(B-1)); repeat at 1 and 14 days with seeds 150001/150014. Blocks do not cross years, gaps retained as missing origins, no iid lead inference, no multiplicity-adjusted claim, descriptive sensitivity on exposed periods. Block lengths frozen before assessment. Report number of blocks and limitations for dependence beyond block length and retrospective selection. No inference on incomplete cohorts.

## Gates and stop

1. Implement explicit retrospective data/cohort adapter; synthetic and historical regression checks. CI-only cohort/hash gate; persist and verify readback.
2. Implement model dispatch and generic independent audit; check source recipe parity without new scored development trials where feasible (synthetic and existing regression tests). Verify backend availability. Freeze machine-readable matrix (80 fits,16 combinations) and workflow command before assessment.
3. Execute only frozen matrix, max six concurrent jobs, 30 minutes/job, workspace target <=2 GB, original stricter 1200/1400s runtime guards preserved; full incumbent explicit 1400s model allowance, no budget relaxation. One infrastructure retry; fix implementation defects with same recipe, no tuning. Log explicit blocked outcomes.
4. Archive all results; run sequential combination assessment and independent auditing; persist final ranking, bootstrap, Pareto, diagnostics, comprehensive report and state, verify durable evidence. No new campaign, station, prospective data or further optimization. CI artifacts have finite retention; preserve aggregated results and source hashes in Git. Stop at approved acceptance, even if no improvement.

No measured outcomes yet; no human action.
