# Local data

Raw and derived datasets are generated artifacts, not committed research results. Use `bash scripts/fetch-ksea.sh 2024 2024` for a small one-year acquisition, or no arguments for the entire declared period. Verify hashes before parsing. Set AEVRAL_DATA_DIR for a persistent cache outside the checkout. Do not use the 2024 sample to tune models (it belongs to the locked test period); use 2018 for development instead. 
