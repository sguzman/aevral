# Forecasting ontology

| Object | Meaning |
|---|---|
| Domain | Family of phenomena, e.g. weather, finance, culture |
| Entity | Measured subject, e.g. station KSEA |
| Variable | Quantity with units and sampling semantics |
| Observation | Entity, variable, event time, value, provenance, quality and availability time (when known) |
| Dataset snapshot | Immutable raw bytes, source query, retrieval time and checksum |
| Forecast origin | Time at which a prediction is issued |
| Information set | Observations available no later than the origin, plus legitimately known future covariates |
| Feature | Reproducible transformation of the information set |
| Target | Quantity to predict at an explicitly defined future valid time |
| Horizon | Target valid time minus origin |
| Example | Features at one origin and future labels, with no information leakage |
| Model | Parameterized mapping from features to predictions |
| Baseline | Reference forecasting procedure with identical information restrictions |
| Split | Chronological partition by target timestamps; purge examples crossing boundaries |
| Experiment | Frozen hypothesis, inputs, model, seed, split, metrics and code/data identities |
| Forecast record | Origin, target times, predictions, model and snapshot identities |
| Resolution | Observed truth and error for a forecast record |

The engine may predict scalars, vectors, probabilities or distributions. Domain adapters handle source-specific units and sampling. No feature uses a future observation, including future-derived imputation or normalization. Fit all transformations on training data alone. Archived event time is not proof of original availability; mark retrospective assumptions explicitly.
