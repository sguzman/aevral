# Continuation checkpoint — phase 7, 2026-11-26

C001 KSEA direct 24-hour temperature forecasting. AEV-001 validation MAE 1.64647975 C (19.699686% skill), results/AEV-001 contains 23 frozen hashes and full metrics. Nix locked shell verified https://github.com/sguzman/aevral/actions/runs/45865004660.

AEV-002 passed https://github.com/sguzman/aevral/actions/runs/45865113357; Rust checks passed 45865113293. Matched 126093 train / 31045 validation cases. MAE 6h=2.12570156,12h=1.77374181,24h=1.65361398,48h=1.59926370,72h=1.59702117; daily=2.05052217. All 256 logged metric records preserved in results/AEV-002. Choose 72h by MAE, 48h essentially tied.

AEV-003/004 calendar comparison implemented (src/windows.rs, experiments/AEV-003-004.md, calendar.yml); pending CI. Compare 72h temperature-only vs UTC hour, annual phase and both, frozen snapshot and matched origins. Verify reference reproduces AEV-002 MAE within 1e-6. Next inspect CI, publish actual calendar results, choose validation winner and implement pressure/dew point/wind feature ingestion and comparisons. Keep test 2023–2025 locked. No local downloads required. Note editing windows.rs also triggers AEV-002 workflow; not needed for new analysis but confirms reference. Future restrict triggers if refactoring. Final freeze and test only after feature comparisons. No human action needed.
