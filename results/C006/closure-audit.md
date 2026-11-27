# C006 closure audit
- 152 unique registry keys (fold,model,seed); all measured, finite, exact cohort sizes.
- 28 unique 2019 screens (first 28 rows); 20 reused finalist screens, 8 nonfinalists. 144 finalist fold/seed outcomes = 12 ×4 deterministic + 8 ×3 ×4 stochastic. All 144 expected keys present, no extras except 8 rejected 2019 screens.
- 59 unique successful benchmark job IDs referenced; archive ingestion checked counts, sample, seed, 24 lead rows per measured outcome and upstream tree counts. All 17 native and 14 upstream annual jobs successful; six-job parallel cap retained.
- 37 including incumbent (12 deterministic +24 stochastic +1 incumbent).
- Scores weighted by origins; 30,138 origins and 723,312 targets per seed. 95 Rust tests in finalist CI. No raw artifact rescore in closure.
- No missing or blocked configurations. Prior CatBoost integration defect and correction recorded separately. No new data.
