# Continuation checkpoint

C001 phase 2, 2026-11-26.

Bootstrap CI passed: https://github.com/sguzman/aevral/actions/runs/45852698484 (f58d9f7). Educational MLP compiled, tests and synthetic smoke passed.

This iteration: UTC minute parser, latest routine report per hourly bin, duplicate handling, missing temperature retention, strict 24+24 examples, chronological target-time split/purge and audit CLI; test fixtures and sample acquisition CI. Request uses repeated data parameters and documented calendar date query. Local environment lacks cargo, Nix and network DNS; use repository CI. New code and actual 2018 sample audit pending CI verification. No weather metrics yet; Nix lock still pending.

Next: inspect both CI runs, record sample checksum and coverage; fix ingestion if required, then implement matched-case baselines and train-only normalization/training loop. Acquire remaining years via persistent cache. Do not tune on 2023–2025. Keep the initial protocol and report negative skill if applicable. Nix lock should be generated with Nix, not fabricated.

Human QA not required at present.
