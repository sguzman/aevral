# AEV-001 — initial validation result

Successful run: https://github.com/sguzman/aevral/actions/runs/45864548360
Code commit: 506df77069ddfb023934bd78942d78db5fb73970
Source: KSEA routine observations, 2000–2022, exact annual hashes in snapshot.sha256. Full measured output in validation.txt. Training 140933 origins; validation 32903 origins. No 2023–2025 data loaded.

| Model | MAE (C) | RMSE (C) |
|---|---:|---:|
| Persistence | 2.98113964 | 4.02513262 |
| Daily persistence | 2.05040264 | 2.70520725 |
| Climatology | 2.58427610 | 3.41777179 |
| MLP | 1.64647975 | 2.19740466 |

MLP reduced aggregate MAE relative to daily persistence by 19.699686%. MLP lead-one MAE 0.62192005 C, lead-24 MAE 1.97459410 C. MLP beats all three baselines at every lead and in all four aggregate seasonal slices. Initial fixed 10-epoch training normalized MSE declined from 0.13463615 to 0.11765849.

Interpretation: local temperature history provides useful predictive information; this is validation evidence, not a locked test result, an operational forecast or proof of superiority to numerical weather prediction. Overlapping examples are correlated; no independent-example confidence claim. Complete-window filtering limits applicability to observed cases.

Next compare historical window lengths on matched origins, then additional observed variables and calendar features. Retain AEV-001 as a reference; no test tuning.
