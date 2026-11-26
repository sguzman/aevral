# AEV-014A — SGD budget and seed sensitivity

[All 18 jobs passed](https://github.com/sguzman/aevral/actions/runs/45877715589), code f9b92bdf91aec32a0815241d555cdf435c3dd31b. All seed-42 reproduction checks passed. Frozen 80 inputs, 121849 training / 30207 validation origins.

| Model | Epochs | Mean MAE C | Sample SD MAE |
|---|---:|---:|---:|
| mlp32 | 10 | 1.39123572 | 0.00650100 |
| mlp32 | 30 | 1.37172515 | 0.00133709 |
| mlp64 | 10 | 1.37411813 | 0.01426593 |
| mlp64 | 30 | 1.35698070 | 0.00939120 |
| mlp32x32 | 10 | 1.38910734 | 0.00605530 |
| mlp32x32 | 30 | 1.36148463 | 0.00618669 |
| rnn | 10 | 1.53576413 | 0.01474428 |
| rnn | 30 | 1.50866325 | 0.00846127 |
| cnn | 10 | 1.55038166 | 0.00545740 |
| cnn | 30 | 1.52054894 | 0.01154279 |
| attention | 10 | 1.55600157 | 0.01821946 |
| attention | 30 | 1.52052335 | 0.01587284 |

36 checkpoints, 1908 full metric records across 18 model/seed files. Each includes RMSE, clipping, timing and per-lead metrics. MLP64 at 30 epochs has lowest mean MAE. All six improve in mean MAE at 30 vs 10 epochs. Three seeds do not establish significance. No holdout access.
