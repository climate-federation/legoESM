# Ocean Regression Matrix

- Generated: 2026-03-02T06:23:44Z
- Overall pass: `True`
- Overall warnings: `True`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C4_L4 | ocean_tests | pass_with_warnings | 0 | 21.8 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C4_L4 | gravity_wave | True | True | ssh_abs_warn | True | True | True | True | 0.00e+00 | 8.87e-08 | -7.13e-09 | 1.05e-03 | 2.00e+01 | 1.11e-01 | 2.00e+00 |
| ocean_tests_C4_L4 | rest_state | True | False | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C4_L4 | wind_gyre | True | False | - | True | True | True | True | 0.00e+00 | 0.00e+00 | -2.72e-14 | 2.62e-04 | 2.00e+01 | 1.47e-06 | 5.00e+00 |

## Logs

- `ocean_tests_C4_L4`: `results/_smoke_ocean_regression_warn/logs/ocean_tests_C4_L4.log`
