# Ocean Regression Matrix

- Generated: 2026-03-02T08:50:45Z
- Overall pass: `True`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | pass | 0 | 64.1 |
| ocean_realistic_C8_L10 | ocean_realistic | pass | 0 | 102.8 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | True | False | - | - | True | True | True | True | 2.53e-08 | 2.53e-08 | 3.28e-08 | 4.42e-04 | 2.00e+01 | 2.22e-02 | 2.00e+00 |
| ocean_tests_C8_L10 | rest_state | True | False | - | - | True | True | True | True | 1.25e-15 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C8_L10 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 1.32e-12 | 4.43e-03 | 2.00e+01 | 6.05e-05 | 5.00e+00 |
| ocean_realistic_C8_L10 | baroclinic_adjustment | True | False | - | - | True | True | True | True | 2.55e-08 | 2.60e-08 | 1.56e-07 | 4.10e-01 | 8.00e+01 | 3.36e+00 | 2.00e+01 |
| ocean_realistic_C8_L10 | kelvin_wave | True | False | - | - | True | True | True | True | 3.51e-08 | 3.51e-08 | 6.67e-09 | 1.76e-04 | 5.00e+00 | 8.85e-03 | 1.00e+00 |
| ocean_realistic_C8_L10 | stommel_gyre | True | False | - | - | True | True | True | True | 2.40e-08 | 2.40e-08 | 4.78e-10 | 5.19e-02 | 1.50e+02 | 3.85e-03 | 1.00e+01 |

## Logs

- `ocean_tests_C8_L10`: `results/_smoke_ocean_regression_x64_realistic_corr10/logs/ocean_tests_C8_L10.log`
- `ocean_realistic_C8_L10`: `results/_smoke_ocean_regression_x64_realistic_corr10/logs/ocean_realistic_C8_L10.log`
