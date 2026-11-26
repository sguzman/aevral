# Frozen finalists, part A (11 fully screened methods)
Selection from the 2019 registry at 62120fcb2a92ad466454331029ac4d7a11c5ca7e; top two per method by MAE, then RMSE (no ties requiring cost), one for OLS. L10 is excluded until full quadratic screen resolves. Freeze these choices independent of L10; do not alter after 2020–22 results. Reuse existing 2019 measurements. Each finalist fits anew on each annual expanding training cohort; compare pooled matched-origin MAE/RMSE including 2019. These are development comparisons, not fresh holdouts.

| Method | Model | 2019 MAE | 2019 RMSE |
|---|---|---:|---:|
| L01 | ols | 1.46768796 | 1.93427561 |
| L02 | ridge-1e-4 | 1.46767092 | 1.93423733 |
| L02 | ridge-1e-6 | 1.46768776 | 1.93427520 |
| L03 | lasso-1e-4 | 1.46776943 | 1.93442307 |
| L03 | lasso-1e-3 | 1.47069685 | 1.93767182 |
| L04 | enet-1e-3-a25 | 1.46814867 | 1.93471868 |
| L04 | enet-1e-3-a75 | 1.46977817 | 1.93663733 |
| L05 | huber-0.25 | 1.46480324 | 1.93582492 |
| L05 | huber-0.5 | 1.46516741 | 1.93388172 |
| L06 | median-1e-4 | 1.46546481 | 1.93801795 |
| L06 | median-0 | 1.46548444 | 1.93805607 |
| L07 | pcr-64 | 1.46788769 | 1.93442285 |
| L07 | pcr-32 | 1.47943917 | 1.94460980 |
| L08 | pls-32 | 1.46782162 | 1.93435761 |
| L08 | pls-16 | 1.48933475 | 1.95309473 |
| L09 | rrr-16 | 1.46770656 | 1.93427248 |
| L09 | rrr-8 | 1.46814017 | 1.93478966 |
| L11 | gam-7-1e-3 | 1.45231366 | 1.90959287 |
| L11 | gam-3-1e-3 | 1.45551824 | 1.91301069 |
| L12 | mars-grid-2-64 | 1.43531178 | 1.88305991 |
| L12 | mars-grid-1-64 | 1.47772393 | 1.94027705 |

63 additional fits across three years, grouped in 18 jobs, max five concurrent while full polynomial runs. Keep individual prediction CSV and metric logs per configuration, explicit blocked outcomes, no retry of nonconvergence by changing recipes. No paid compute or test period access.
