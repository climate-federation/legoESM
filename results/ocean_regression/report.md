# Ocean Regression Matrix

- Generated: 2026-03-02T05:54:06Z
- Overall pass: `True`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | pass | 0 | 22.5 |
| ocean_tests_C16_L20 | ocean_tests | pass | 0 | 23.4 |
| ocean_realistic_C8_L10 | ocean_realistic | pass | 0 | 26.5 |

## Case Checks

| Run | Case | Pass | all_finite | land_zero | heat | salt | eta |
|---|---|---|---|---|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | True | True | True | 2.99e-07 | -4.47e-07 | -7.69e-08 |
| ocean_tests_C8_L10 | rest_state | True | True | True | -7.48e-08 | 0.00e+00 | 0.00e+00 |
| ocean_tests_C8_L10 | wind_gyre | True | True | True | -3.13e-07 | 1.61e-06 | 4.31e-07 |
| ocean_tests_C16_L20 | gravity_wave | True | True | True | 5.97e-07 | 1.79e-07 | -1.13e-07 |
| ocean_tests_C16_L20 | rest_state | True | True | True | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| ocean_tests_C16_L20 | wind_gyre | True | True | True | -2.09e-07 | 0.00e+00 | 1.81e-07 |
| ocean_realistic_C8_L10 | baroclinic_adjustment | True | True | True | -3.58e-07 | -2.06e-06 | 1.15e-06 |
| ocean_realistic_C8_L10 | kelvin_wave | True | True | True | -9.38e-07 | -8.04e-07 | 4.67e-08 |
| ocean_realistic_C8_L10 | stommel_gyre | True | True | True | -1.04e-06 | -1.61e-06 | 9.78e-08 |

## Logs

- `ocean_tests_C8_L10`: `results/ocean_regression/logs/ocean_tests_C8_L10.log`
- `ocean_tests_C16_L20`: `results/ocean_regression/logs/ocean_tests_C16_L20.log`
- `ocean_realistic_C8_L10`: `results/ocean_regression/logs/ocean_realistic_C8_L10.log`
