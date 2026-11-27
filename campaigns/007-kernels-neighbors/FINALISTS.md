# C007 frozen finalists

Selected from all 41 measured 2019 screens by manifest MAE, RMSE, then compute cost. No boundary ties required the cost tiebreaker. Frozen recipes; no new data, no tuning on 2020–22. 2019 is development, not an untouched holdout.

| Family | Configuration | Screen seed | 2019 MAE °C | 2019 RMSE °C |
|---|---|---|---:|---:|
| K01 | knn-k16 | deterministic | 1.59227390 | 2.06374566 |
| K01 | knn-k32 | deterministic | 1.60847813 | 2.08038736 |
| K02 | analog-k16-inverse | deterministic | 1.58604072 | 2.05652652 |
| K02 | analog-k16-gaussian | deterministic | 1.58769915 | 2.05840897 |
| K03 | season-k16-d30 | deterministic | 1.65237903 | 2.17127893 |
| K03 | season-k8-d30 | deterministic | 1.65894053 | 2.17703494 |
| K04 | local-k128-l01 | deterministic | 1.37195009 | 1.82169663 |
| K04 | local-k64-l01 | deterministic | 1.41022498 | 1.88008412 |
| K05 | linear-krr-l001 | deterministic | 1.51224494 | 1.97731958 |
| K05 | linear-krr-l01 | deterministic | 1.68779093 | 2.19417262 |
| K06 | rbf-krr-s1-l1 | deterministic | 1.39084668 | 1.84799529 |
| K06 | rbf-krr-s1-l01 | deterministic | 1.46211482 | 1.95336877 |
| K07 | svr-s1-c1 | deterministic | 1.40649144 | 1.87076562 |
| K07 | svr-s1-c10 | deterministic | 1.41467660 | 1.89047014 |
| K08 | gp-s1 | deterministic | 1.44283129 | 1.90324250 |
| K08 | gp-s05 | deterministic | 1.58191727 | 2.10234541 |
| K09 | fitc-s1-m256 | deterministic | 1.37723166 | 1.82825816 |
| K09 | fitc-s1-m128 | deterministic | 1.41968820 | 1.87521576 |
| K10 | rff-s1-m256 | 42 | 1.88266398 | 2.45313193 |
| K10 | rff-s1-m128 | 42 | 1.92873945 | 2.51186785 |
| K11 | nystrom-s1-m256 | deterministic | 1.84759723 | 2.41470335 |
| K11 | nystrom-s1-m128 | deterministic | 1.85517621 | 2.42631308 |

## Additional annual evaluation matrix

20 deterministic finalists × 2020–22 = 60 fits. Two K10 finalists × (2019 seeds 123/2026 plus 2020–22 seeds 42/123/2026) = 22 fits. Total 82 new outcomes; reuse all existing 2019 screens. Keep max six parallel jobs, 30 minutes/job, and frozen guards. Report seed-specific four-year pooled scores and sample SD across three RFF seeds. No annual finalist runs launched at this checkpoint.

Screen registry: [registry.csv](../../results/C007/registry.csv). Final batch [CI](https://github.com/sguzman/aevral/actions/runs/45921413274).
