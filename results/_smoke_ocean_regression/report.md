# Ocean Regression Matrix

- Generated: 2026-03-02T05:43:16Z
- Overall pass: `True`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C4_L4 | ocean_tests | pass | 0 | 21.9 |

## Case Checks

| Run | Case | Pass | all_finite | land_zero | heat | salt | eta |
|---|---|---|---|---|---:|---:|---:|
| ocean_tests_C4_L4 | gravity_wave | True | True | True | 0.00e+00 | 8.87e-08 | -7.13e-09 |
| ocean_tests_C4_L4 | rest_state | True | True | True | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| ocean_tests_C4_L4 | wind_gyre | True | True | True | 0.00e+00 | 0.00e+00 | 1.15e-13 |

## Logs

- `ocean_tests_C4_L4`: `results/_smoke_ocean_regression/logs/ocean_tests_C4_L4.log`
