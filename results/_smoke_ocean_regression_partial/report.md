# Ocean Regression Matrix

- Generated: 2026-03-02T06:32:09Z
- Overall pass: `True`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C4_L4 | ocean_tests | pass | 0 | 21.7 |
| ocean_realistic_C4_L4 | ocean_realistic | pass | 0 | 22.5 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C4_L4 | gravity_wave | True | False | - | - | True | True | True | True | 0.00e+00 | 8.87e-08 | -7.13e-09 | 7.95e-04 | 2.00e+01 | 1.11e-01 | 2.00e+00 |
| ocean_tests_C4_L4 | rest_state | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.00e-01 | 0.00e+00 | 1.00e-01 |
| ocean_tests_C4_L4 | wind_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | -2.72e-14 | 2.62e-04 | 2.00e+01 | 1.47e-06 | 5.00e+00 |
| ocean_realistic_C4_L4 | baroclinic_adjustment | True | False | - | - | True | True | True | True | -8.91e-08 | -8.87e-08 | -5.09e-10 | 6.94e-02 | 8.00e+01 | 1.43e-01 | 2.00e+01 |
| ocean_realistic_C4_L4 | kelvin_wave | True | False | - | - | True | True | True | True | 1.04e-07 | 0.00e+00 | 2.55e-10 | 1.80e-04 | 5.00e+00 | 2.66e-02 | 1.00e+00 |
| ocean_realistic_C4_L4 | stommel_gyre | True | False | - | - | True | True | True | True | 0.00e+00 | 0.00e+00 | -1.20e-13 | 2.72e-03 | 1.50e+02 | 1.35e-04 | 1.00e+01 |

## Logs

- `ocean_tests_C4_L4`: `results/_smoke_ocean_regression_partial/logs/ocean_tests_C4_L4.log`
- `ocean_realistic_C4_L4`: `results/_smoke_ocean_regression_partial/logs/ocean_realistic_C4_L4.log`
