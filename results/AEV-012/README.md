# AEV-012 convolution screen
[Successful CI](https://github.com/sguzman/aevral/actions/runs/45877397897), code c1141d402024e3912d0daaefc158f6f3b3bef68e. Frozen 80 inputs and matched 121849 training /30207 validation origins.
| Model | Parameters | MAE C | RMSE C | Train seconds |
|---|---:|---:|---:|---:|
| CNN | 3072 | 1.54584788 | 2.03486604 | 22.534 |
| TCN | 3072 | 1.56940238 | 2.06598677 | 21.784 |

Identical capacity and recipe; ordinary dilation [1,1,1] beats [1,2,4] here. Both worse than MLP64 1.38652827 and original MLP32 1.39610378. All 124 metric records preserved. These are compact fixed-budget variants, not evidence that the architecture families cannot improve. Test holdout untouched.
