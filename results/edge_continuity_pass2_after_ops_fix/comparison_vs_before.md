# Edge Continuity Comparison (After Operator Patch)

Before: `results/edge_continuity_pass2/edge_continuity_summary.json`
After: `results/edge_continuity_pass2_after_ops_fix/edge_continuity_summary.json`

| Case | Step | Field | rel_p99 before | rel_p99 after | rel_p99 delta % | frac_rel_gt5 before | frac_rel_gt5 after | jump_p99 before | jump_p99 after |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| NH TC2a | 0 | rho_prime | 0.000 | 0.000 | +0.0% | 0.000 | 0.000 | 0.000000e+00 | 0.000000e+00 |
| NH TC2a | 0 | w_mid | 0.000 | 0.000 | +0.0% | 0.000 | 0.000 | 0.000000e+00 | 0.000000e+00 |
| NH TC2a | 0 | wind_low | 1.055 | 1.055 | +0.0% | 0.000 | 0.000 | 6.167879e-01 | 6.167879e-01 |
| NH TC2a | 450 | rho_prime | 9.479 | 11.692 | +23.4% | 0.231 | 0.222 | 7.699014e-03 | 7.694781e-03 |
| NH TC2a | 450 | w_mid | 7.594 | 7.701 | +1.4% | 0.130 | 0.056 | 5.675057e-01 | 5.651356e-01 |
| NH TC2a | 450 | wind_low | 7.658 | 7.702 | +0.6% | 0.019 | 0.019 | 1.233041e+00 | 1.213649e+00 |
| NH TC2a | 900 | rho_prime | 10.740 | 11.505 | +7.1% | 0.056 | 0.042 | 1.167008e-02 | 1.161908e-02 |
| NH TC2a | 900 | w_mid | 7.931 | 7.559 | -4.7% | 0.148 | 0.037 | 3.584930e+00 | 3.561362e+00 |
| NH TC2a | 900 | wind_low | 3.684 | 2.905 | -21.1% | 0.009 | 0.009 | 1.530184e+00 | 1.453693e+00 |
| SW Williamson 2 | 0 | height | 1.045 | 1.045 | +0.0% | 0.000 | 0.000 | 8.308667e+01 | 8.308667e+01 |
| SW Williamson 2 | 0 | wind_speed | 1.055 | 1.055 | +0.0% | 0.000 | 0.000 | 1.190733e+00 | 1.190733e+00 |
| SW Williamson 2 | 480 | height | 18.504 | 5.830 | -68.5% | 0.106 | 0.023 | 9.336206e+01 | 9.735632e+01 |
| SW Williamson 2 | 480 | wind_speed | 3.407 | 3.303 | -3.0% | 0.000 | 0.000 | 1.552107e+01 | 1.612046e+01 |
| SW Williamson 2 | 960 | height | 14.494 | 22.942 | +58.3% | 0.088 | 0.097 | 1.428362e+02 | 1.130371e+02 |
| SW Williamson 2 | 960 | wind_speed | 4.024 | 3.655 | -9.2% | 0.005 | 0.005 | 1.270136e+01 | 1.251925e+01 |
