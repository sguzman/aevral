# KSEA 2018 sample audit

Verified CI: https://github.com/sguzman/aevral/actions/runs/45864390584
Code: cf3b2419b0f11f4c8a1d637f0feaa666b1de122f

| Metric | Count |
|---|---:|
| Unique routine reports | 8740 |
| Duplicate rows | 0 |
| Hourly bins | 8759 |
| Absent hourly reports | 20 |
| Missing temperature in selected reports | 0 |
| Eligible 24+24 windows | 8060 |
| Excluded for missingness | 652 |
| Excluded for split boundary | 0 |

Checksum verification passed; exact hash stored in the run's ksea-2018-audit artifact (not printed in log). All windows are training-only. No weather prediction scores from this run. First and last partial bins excluded by the audit range.
