# KSEA origin-time covariate audit

[Successful CI](https://github.com/sguzman/aevral/actions/runs/45865568542), code 3f43e88dd08d645c75912479c49a9f928e9c29db. Frozen 2000–2022 snapshot; no test access. Temperature timestamps and values exactly align with original parser. No malformed or broad out-of-range values in dew point, pressure, direction or speed (zero in each counter).

| Split | Complete temperature | Dew point | Altimeter | Wind vector | All three |
|---|---:|---:|---:|---:|---:|
| Train | 126093 | 126066 | 126087 | 121882 | 121849 |
| Validation | 31045 | 31045 | 31045 | 30207 | 30207 |

Use joint eligible cases for next feature comparisons. Wind availability limits the joint cohort. Reference must be retrained on the same cases, not compared directly against AEV-003/004 aggregate score. These are coverage counts, not prediction results.
