# Continuation checkpoint

Bootstrap phase, 2026-11-26.

Scope: general forecasting research; C001 neural weather. Rust, Nix, CPU-first educational MLP. KSEA selected from documented IEM station metadata.

Implemented in bootstrap: ontology, protocol, dataset acquisition script, simple MLP with synthetic smoke command and unit tests. No external data acquired, no real forecast metrics. Build/test status pending actual execution; Nix lockfile not generated.

Next unfinished portion: verify Rust tests, lock the Nix flake, acquire a small annual sample using the cache script, build/test the UTC binning and missingness audit, then extend acquisition and execute baseline/MLP experiments. Do not tune on test. Store actual run evidence and update this checkpoint. Avoid downloading already verified cache files.

Human QA: not required for documentation bootstrap; local execution may be needed if no CI runner is available.
