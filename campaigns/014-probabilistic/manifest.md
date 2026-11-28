# C014 — Frozen probabilistic forecasting protocol

## Scope and budget

24 predeclared recipes × two annual assessments = 48 outcomes maximum. 20 distribution-valued recipes and four interval-only conformal recipes. Use C013 verified joined source artifact 15239944097 (run 46089681073, code d5f37c4cc7742875ca1631be9a4533210dede85a), with all six SHA256 values in results/C013/source-audit.txt. No new base-model fits, raw observations, 2023–26 labels, external data or additional trial configurations. Compute centers I (full-training incumbent) and B (equal mean of I,L,R,M, C013 E08) directly from joined columns, not from assessment targets. Ensemble spread d=max(I,L,R,M)-min(I,L,R,M) is observable at forecast origin.

Calibrate 2021 only on 2020 out-of-fold forecasts, 2022 on 2020–2021 out-of-fold forecasts. All target windows lie in their own years; no assessment targets enter the fit for that year. Assess on identical 7,501/7,639 origins and all 24 leads (363,360 scalar targets). These are retrospective development results: B was the C013 development winner on these same assessment years, so its inclusion is post-selection and cannot establish independent skill. No further selection/tuning during this campaign. Use lead-specific calibration except where explicitly global. Celsius throughout.

## Frozen matrix

| ID | Family | Center | Distribution or interval construction |
|---|---|---|---|
| P01 | Empirical residual | I | Global signed residual sample |
| P02 | Empirical residual | B | Global signed residual sample |
| P03 | Empirical residual | I | Per-lead signed residual sample |
| P04 | Empirical residual | B | Per-lead signed residual sample |
| P05 | Gaussian head | I | Per-lead residual mean and SD |
| P06 | Gaussian head | B | Per-lead residual mean and SD |
| P07 | Gaussian head | B | Per-lead, two spread regimes |
| P08 | Student-t head | B | Per-lead, fixed df=3 |
| P09 | Student-t head | B | Per-lead, two spread regimes, df=3 |
| P10 | Quantile regression | I | Per-lead affine spread-conditioned residual quantiles |
| P11 | Quantile regression | B | Per-lead affine spread-conditioned residual quantiles |
| P12 | Distributional tree | I | Per-lead depth-2 residual distribution tree |
| P13 | Distributional tree | B | Per-lead depth-2 residual distribution tree |
| P14 | Empirical analog | I | Per-lead 128 nearest center-temperature analogs |
| P15 | Empirical analog | B | Per-lead 128 nearest center-temperature analogs |
| P16 | Empirical analog | B | Per-lead 128 nearest spread analogs |
| P17 | Ensemble distribution | — | Raw equal-mass I,L,R,M member forecasts |
| P18 | Ensemble distribution | — | Raw equal-mass all eight member forecasts |
| P19 | Ensemble distribution | — | Four members plus pooled per-lead member residual distribution |
| P20 | Ensemble distribution | — | Four members plus member-specific per-lead residual distributions |
| P21 | Split conformal | I | Per-lead symmetric absolute residual |
| P22 | Split conformal | B | Per-lead symmetric absolute residual |
| P23 | Split conformal | B | Per-lead asymmetric one-sided residual |
| P24 | Split conformal | B | Per-lead normalized absolute residual, scale=max(0.25,d) |

## Exact distribution definitions

For all distribution recipes issue a finite, equally weighted predictive distribution, with its sorted support (including repeats). P01–P16 issue 99 support values at p_j=(j+0.5)/99, j=0..98. Empirical residual quantiles use inverse empirical CDF order statistic ceil(p*n), clamped 1..n. Residual r=y-center. P01–P04 add sampled signed residual quantiles to center. Parametric heads P05–P09 are postprocessors, not jointly trained neural distribution heads: fit population residual mean and SD with minimum SD 0.05°C; Gaussian uses mean+SD*Phi^-1(p), Student-t uses mean+SD/sqrt(3)*t3^-1(p) (variance matched). Invert standard normal CDF and t3 CDF by deterministic bisection with absolute error <=1e-8 in standardized quantile; t3 CDF = 1/2 + [atan(z)+z/(1+z²)]/pi with z=t/sqrt(3). Conditional heads split each lead by its calibration median d, ties low; fewer than 128 rows in group uses same-lead unconditional estimate. Record thresholds, counts, fitted moments and fallback. These issued discrete distributions approximate the continuous parametric heads, not exact continuous Gaussian/t CRPS.

P10–P11: for each lead, fit 19 residual quantile curves at q=0.05,0.10,...,0.95 using affine rhat=a+b*d. Finite slope grid b in {-2,-1,-0.5,-0.25,0,0.25,0.5,1,2}; for each slope estimate intercept by inverse empirical CDF of r-b*d and minimize total calibration pinball at q. Ties choose smallest absolute slope, then smaller signed slope. This is constrained linear quantile regression, not an unconstrained optimizer. Efficiently sort adjusted residuals once per slope and use prefix sums for loss. At inference compute 19 fitted quantiles, rearrange ascending to prevent crossing, linearly interpolate in probability to 99 support quantiles, clamp tails to outer fitted values; report crossing/rearrangement incidence. Store slopes/intercepts/losses. Tail truncation is a known limitation.

P12–P13: greedy depth-2 regression tree on signed residual distributions, separately by lead. Candidate split features center and spread, thresholds empirical quartiles 0.25,0.50,0.75 in current node, ties low. Minimum child size 128. Minimize weighted within-child empirical CRPS evaluated on calibration residuals (equivalent to half average pairwise absolute difference of residuals, computed from sorted residuals); compare against parent, split only for strict improvement >1e-12, tie by feature order center then spread and ascending quartile. Leaves store 99 inverse-CDF residual quantiles. This is a distributional regression tree over forecast features, not a raw-weather boosted quantile forest. Log node split/leaf counts, objective and support.

P14–P16: for each lead sort calibration rows by selected scalar (center or spread), then origin for tie-breaking. Choose 128 nearest rows by absolute scalar distance, tie by origin ascending; use their signed residual inverse-CDF quantiles. Use ordered search plus deterministic boundary tie handling (including identical scalar values). Do not access later origins or targets. This is a one-dimensional analog distribution, not full 80-feature nearest-neighbor weather retrieval.

P17/P18: exact discrete empirical ensemble distributions over four or eight raw members, equal weights. P19: form pooled residual sample across four member predictions I,L,R,M per lead, sample 99 quantiles; issue 396 support values (each of four member predictions plus each of 99 residual quantiles). P20: same but per-member residual sample. All equally weighted; no assumption of independent ensemble members or joint temporal path distribution. Record fitted residual supports. These are scalar marginal forecasts, not coherent multivariate 24-hour trajectory samples.

## Conformal intervals

For P21/P22 use per-lead absolute residual scores; P24 uses |r|/max(0.25,d). For central nominal coverage c in {0.50,0.80,0.90,0.95}, score quantile at rank min(n,ceil((n+1)*c)); return center ± score (multiply by inference scale for P24). For P23 use upper scores r and lower scores -r independently at rank min(n,ceil((n+1)*(1+c)/2)); interval [B-q_lower,B+q_upper]. Do not clip signed one-sided thresholds; assert nested, non-inverted intervals or explicit numerical block. Center is the point median proxy for interval-only recipes, not a separately fitted conditional median. No iid exchangeability or finite-sample coverage guarantee for serially dependent observations. No CRPS or quantile pinball ranking for interval-only methods; report separate interval score and coverage. Fitted thresholds and sample counts retained.

## Scoring

For distribution-valued recipes sorted support x_1..x_m, equal masses: exact CRPS = mean|x_i-y| - sum_i(2*i-m-1)*x_i/m² (1-based i). Score pinball at q={0.025,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.975} using issued inverse-CDF quantile; point MAE/RMSE at q=0.50. Central intervals c={0.50,0.80,0.90,0.95} use lower/upper inverse CDF at (1-c)/2,(1+c)/2. For every recipe report inclusive empirical coverage, mean width, and interval score width+2/(1-c)*(max(lower-y,0)+max(y-upper,0)). For conformal use center proxy for point MAE/RMSE; no CRPS/pinball. Report mean CRPS primary for full distributions, mean pinball secondary, interval score separately per nominal level for all recipes. Coverage alone is not skill; report coverage error and sharpness together. Summaries overall, per lead, per month and fold (pool by actual scalar count; RMSE pooled squared). Report computation time, peak RSS, source center costs separately; no inference latency claims from cached forecasts. Export support or compact exact reconstruction evidence plus issued quantiles/intervals, target, origin/lead, CRPS/pinball contributions; independent metric/reconstruction checks. Preserve coefficients/tree/analog configuration and source hashes, aggregate results and all failure records. Bulk predictions finite-retention CI artifacts. Rust-first; synthetic tests for scoring, quantiles, parametric inverse, residual fit, tree/analog ties, conformal ranks and chronological guards. No evaluation before verified implementation. Stop at 48 outcomes; then C015.
