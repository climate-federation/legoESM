# Ocean Regression Matrix

- Generated: 2026-03-02T08:44:33Z
- Overall pass: `True`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | pass | 0 | 27.5 |
| ocean_realistic_C8_L10 | ocean_realistic | pass | 0 | 44.8 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | True | False | - | - | True | True | True | True | 2.24e-07 | -2.68e-07 | 7.18e-09 | 4.38e-04 | 2.00e+01 | 2.23e-02 | 2.00e+00 |
| ocean_tests_C8_L10 | rest_state | True | False | - | - | True | True | True | True | 1.50e-07 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C8_L10 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 8.97e-12 | 4.43e-03 | 2.00e+01 | 6.07e-05 | 5.00e+00 |
| ocean_realistic_C8_L10 | baroclinic_adjustment | True | False | - | - | True | True | True | True | -5.37e-07 | -6.26e-07 | -2.71e-07 | 4.37e-01 | 8.00e+01 | 3.56e+00 | 2.00e+01 |
| ocean_realistic_C8_L10 | kelvin_wave | True | False | - | - | True | True | True | True | -9.38e-07 | -7.15e-07 | 2.41e-08 | 5.22e-05 | 5.00e+00 | 5.96e-03 | 1.00e+00 |
| ocean_realistic_C8_L10 | stommel_gyre | True | False | - | - | True | True | True | True | -4.17e-07 | -8.94e-08 | 7.43e-10 | 8.65e-02 | 1.50e+02 | 4.14e-03 | 1.00e+01 |

## Logs

- `ocean_tests_C8_L10`: `results/_smoke_ocean_regression_realistic_corr8/logs/ocean_tests_C8_L10.log`
- `ocean_realistic_C8_L10`: `results/_smoke_ocean_regression_realistic_corr8/logs/ocean_realistic_C8_L10.log`
