# AEV-014B — Adam sensitivity
[All 18 jobs passed](https://github.com/sguzman/aevral/actions/runs/45878735829), code 08b51e92ba7047e747b7ad0654de38844fe13e36. Frozen cohort and three seeds, same architecture, 10/30 epochs. 1908 full metric records. Below: all 24 configuration means across A and B, ranked by mean validation MAE.

| Optimizer | Model | Epochs | Mean MAE C | Sample SD | Mean RMSE C |
|---|---|---:|---:|---:|---:|
| sgd | mlp64 | 30 | 1.35698070 | 0.00939120 | 1.82018595 |
| sgd | mlp32x32 | 30 | 1.36148463 | 0.00618669 | 1.82450495 |
| sgd | mlp32 | 30 | 1.37172515 | 0.00133709 | 1.83464505 |
| sgd | mlp64 | 10 | 1.37411813 | 0.01426593 | 1.84358281 |
| sgd | mlp32x32 | 10 | 1.38910734 | 0.00605530 | 1.85690894 |
| sgd | mlp32 | 10 | 1.39123572 | 0.00650100 | 1.86050037 |
| sgd | rnn | 30 | 1.50866325 | 0.00846127 | 2.00297575 |
| sgd | attention | 30 | 1.52052335 | 0.01587284 | 2.01394070 |
| sgd | cnn | 30 | 1.52054894 | 0.01154279 | 2.01556842 |
| adam | mlp32 | 30 | 1.52243045 | 0.02352660 | 2.01300292 |
| adam | mlp64 | 30 | 1.52418987 | 0.00889204 | 2.01125803 |
| adam | mlp32 | 10 | 1.53316249 | 0.02919158 | 2.02259626 |
| adam | mlp32x32 | 30 | 1.53472472 | 0.03425888 | 2.03387580 |
| sgd | rnn | 10 | 1.53576413 | 0.01474428 | 2.03169449 |
| adam | mlp64 | 10 | 1.53581093 | 0.00661909 | 2.02479357 |
| adam | mlp32x32 | 10 | 1.53935666 | 0.02995754 | 2.03838234 |
| adam | attention | 30 | 1.54719258 | 0.01503098 | 2.04999576 |
| sgd | cnn | 10 | 1.55038166 | 0.00545740 | 2.04285370 |
| adam | rnn | 30 | 1.55053017 | 0.01468404 | 2.05332996 |
| sgd | attention | 10 | 1.55600157 | 0.01821946 | 2.05614755 |
| adam | attention | 10 | 1.56942422 | 0.02785720 | 2.06971636 |
| adam | rnn | 10 | 1.57031260 | 0.01987490 | 2.07083006 |
| adam | cnn | 30 | 1.58739424 | 0.01387671 | 2.09308561 |
| adam | cnn | 10 | 1.61685987 | 0.02316322 | 2.12239952 |

Winner per preregistered rule: MLP64, SGD .005, 30 epochs. Adam .001 loses for each matched architecture/epoch combination in mean MAE. This compares two recipes, not optimally tuned optimizer families. Three seeds do not establish significance. All new holdout data remain unopened.
