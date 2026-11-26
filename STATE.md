# Continuation checkpoint — 2026-11-26, phase 5

C001 general forecasting, KSEA 24-hour direct temperature MLP. AEV-001 validated successfully (https://github.com/sguzman/aevral/actions/runs/45864548360), code 506df77069ddfb023934bd78942d78db5fb73970. 23 years acquired and fingerprinted, 140933 training and 32903 validation windows. MLP MAE 1.64647975 C, daily baseline 2.05040264 C, skill 19.699686%. Results, exact hashes and full logged metrics committed under results/AEV-001. Test years untouched. Nix CI succeeded (https://github.com/sguzman/aevral/actions/runs/45864667990); actual generated lock committed, subsequent verification pending.

Next unfinished work: AEV-002 historical input windows 6/12/24/48/72 hours on identical eligible train/validation origins (daily baseline requires 24 hours; match all variants to 72-hour complete history). Preserve AEV-001 original results and use same fixed recipe with dimension-adjusted initialization; record matched-origin counts and model results. Then calendar and other observed features as planned. Validate only 2000–2022, check annual hashes against AEV-001, reuse cached files. No test tuning. After selected model freeze, final locked test evaluation and report. Do not invent additional campaigns before completing this milestone.

Human QA not needed. No local build or data download required.

## Phase 6

AEV-002 protocol and matched-window comparison implementation added (src/windows.rs, experiments/AEV-002.md, workflow windows.yml). New code pending CI, uses AEV-001 frozen hashes and shared cache; no test data. Next inspect Rust and AEV-002 CI, publish actual metrics and choose window by validation MAE, then compare calendar and meteorological inputs on matched origins. Post-lock Nix run 45865004660 pending at last check. No human action required.
