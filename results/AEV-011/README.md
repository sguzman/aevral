# AEV-011 recurrent screen
[Successful CI](https://github.com/sguzman/aevral/actions/runs/45877263961), code 0d1a5a5ac3f526ee33eb0ae84fd4b72360283cfc. Same 121849 training and 30207 validation origins, inputs and normalization.
| Model | Parameters | MAE C | RMSE C | Train seconds |
|---|---:|---:|---:|---:|
| RNN | 968 | 1.52718775 | 2.02040131 | 14.123 |
| GRU | 1704 | 1.53676733 | 2.03309684 | 31.033 |
| LSTM | 2072 | 1.56212105 | 2.06542693 | 55.797 |

MLP64 1.38652827, MLP32 1.39610378, daily 2.05315123. Recurrent models underperform dense models in this screen. Smaller capacity, final-state bottleneck, six-hour patches, fixed optimizer and seed prevent broad family-level conclusions. Timing is runner-specific. Full logs per architecture preserved, including clipping counts. New holdout untouched.
