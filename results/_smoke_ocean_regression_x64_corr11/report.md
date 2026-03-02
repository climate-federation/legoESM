# Ocean Regression Matrix

- Generated: 2026-03-02T08:52:47Z
- Overall pass: `True`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | pass | 0 | 35.1 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | True | False | - | - | True | True | True | True | 2.52e-08 | 2.52e-08 | 1.95e-08 | 2.24e-03 | 2.00e+01 | 4.98e-02 | 2.00e+00 |
| ocean_tests_C8_L10 | rest_state | True | False | - | - | True | True | True | True | 1.25e-15 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C8_L10 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | -5.77e-12 | 3.33e-03 | 2.00e+01 | 4.92e-05 | 5.00e+00 |

## Logs

- `ocean_tests_C8_L10`: `results/_smoke_ocean_regression_x64_corr11/logs/ocean_tests_C8_L10.log`
