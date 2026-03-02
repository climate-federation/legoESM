# Ocean Regression Matrix

- Generated: 2026-03-02T08:46:19Z
- Overall pass: `False`
- Overall warnings: `False`
- Thresholds: |heat drift| <= 1.00e-03, |salt drift| <= 1.00e-03, |mean eta drift| <= 1.00e-03 m, speed <= 2.00e+02 m/s, |SSH| <= 5.00e+01 m
- Case profiles enabled: `True`

## Run Status

| Run | Suite | Status | Return | Wall (s) |
|---|---|---|---:|---:|
| ocean_tests_C8_L10 | ocean_tests | command_failed | 1 | 35.8 |

## Case Checks

| Run | Case | Pass | Warn | Warnings | Error | finite | land | speed_ok | ssh_ok | heat | salt | eta | speed | speed_lim | ssh_abs | ssh_lim |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ocean_tests_C8_L10 | gravity_wave | False | False | - | TypeError: scan body function carry input and carry output must have equal types, but they differ:  The input carry component loop_carry[1][0] has type float32[6,8,8] but the corresponding output carry component has type float64[6,8,8], so the dtypes do not match.  Revise the function so that all output types match the corresponding input types. | False | False | False | False | n/a | n/a | n/a | n/a | 2.00e+01 | n/a | 2.00e+00 |
| ocean_tests_C8_L10 | rest_state | False | False | - | TypeError: scan body function carry input and carry output must have equal types, but they differ:  The input carry component loop_carry[1][0] has type float32[6,8,8] but the corresponding output carry component has type float64[6,8,8], so the dtypes do not match.  Revise the function so that all output types match the corresponding input types. | False | False | False | False | n/a | n/a | n/a | n/a | 1.00e-01 | n/a | 1.00e-01 |
| ocean_tests_C8_L10 | wind_gyre | False | False | - | TypeError: scan body function carry input and carry output must have equal types, but they differ:  The input carry component loop_carry[1][0] has type float32[6,8,8] but the corresponding output carry component has type float64[6,8,8], so the dtypes do not match.  Revise the function so that all output types match the corresponding input types. | False | False | False | False | n/a | n/a | n/a | n/a | 2.00e+01 | n/a | 5.00e+00 |

## Logs

- `ocean_tests_C8_L10`: `results/_smoke_ocean_regression_x64_corr9/logs/ocean_tests_C8_L10.log`
