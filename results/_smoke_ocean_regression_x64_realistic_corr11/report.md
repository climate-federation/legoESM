# Ocean Regression Matrix

- Generated: 2026-03-02T09:54:21Z
- Overall pass: `True`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | pass | 0 | 14.4 |
| ocean_tests_C16_L20 | ocean_tests | pass | 0 | 14.6 |
| ocean_realistic_C8_L10 | ocean_realistic | pass | 0 | 17.4 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | True | False | - | - | True | True | True | True | -9.72e-07 | -8.94e-07 | 2.46e-08 | 6.70e+00 | 2.00e+01 | 1.37e-01 | 2.00e+00 |
| ocean_tests_C8_L10 | rest_state | True | False | - | - | True | True | True | True | 7.48e-08 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C8_L10 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 8.09e-11 | 3.45e+00 | 2.00e+01 | 3.99e-04 | 5.00e+00 |
| ocean_tests_C16_L20 | gravity_wave | True | False | - | - | True | True | True | True | -4.47e-07 | -2.60e-06 | 1.38e-07 | 1.01e+01 | 2.00e+01 | 8.04e-01 | 2.00e+00 |
| ocean_tests_C16_L20 | rest_state | True | False | - | - | True | True | True | True | 7.46e-08 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C16_L20 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | -2.29e-10 | 2.29e+00 | 2.00e+01 | 6.69e-04 | 5.00e+00 |
| ocean_realistic_C8_L10 | baroclinic_adjustment | True | False | - | - | True | True | True | True | -3.13e-06 | -1.16e-06 | 1.38e-06 | 4.30e+01 | 8.00e+01 | 1.26e+01 | 2.00e+01 |
| ocean_realistic_C8_L10 | kelvin_wave | True | False | - | - | True | True | True | True | -1.04e-06 | -3.57e-07 | 5.13e-10 | 7.57e-02 | 5.00e+00 | 5.60e-03 | 1.00e+00 |
| ocean_realistic_C8_L10 | stommel_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | -2.50e-06 | 3.71e-08 | 9.69e+01 | 1.50e+02 | 3.40e+00 | 1.00e+01 |

## Logs

- `ocean_tests_C8_L10`: `results/_smoke_ocean_regression_x64_realistic_corr11/logs/ocean_tests_C8_L10.log`
- `ocean_tests_C16_L20`: `results/_smoke_ocean_regression_x64_realistic_corr11/logs/ocean_tests_C16_L20.log`
- `ocean_realistic_C8_L10`: `results/_smoke_ocean_regression_x64_realistic_corr11/logs/ocean_realistic_C8_L10.log`
