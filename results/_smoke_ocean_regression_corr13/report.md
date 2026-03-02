# Ocean Regression Matrix

- Generated: 2026-03-02T09:19:44Z
- Overall pass: `True`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | pass | 0 | 34.6 |
| ocean_tests_C16_L20 | ocean_tests | pass | 0 | 148.7 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | True | False | - | - | True | True | True | True | 2.99e-07 | -4.47e-07 | -7.69e-08 | 6.53e+00 | 2.00e+01 | 1.15e-01 | 2.00e+00 |
| ocean_tests_C8_L10 | rest_state | True | False | - | - | True | True | True | True | -7.48e-08 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C8_L10 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 7.56e-11 | 3.45e+00 | 2.00e+01 | 3.99e-04 | 5.00e+00 |
| ocean_tests_C16_L20 | gravity_wave | True | False | - | - | True | True | True | True | 3.13e-06 | 2.69e-07 | -8.01e-08 | 7.81e+00 | 2.00e+01 | 6.53e-01 | 2.00e+00 |
| ocean_tests_C16_L20 | rest_state | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C16_L20 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | -2.46e-10 | 2.29e+00 | 2.00e+01 | 6.69e-04 | 5.00e+00 |

## Logs

- `ocean_tests_C8_L10`: `results/_smoke_ocean_regression_corr13/logs/ocean_tests_C8_L10.log`
- `ocean_tests_C16_L20`: `results/_smoke_ocean_regression_corr13/logs/ocean_tests_C16_L20.log`
