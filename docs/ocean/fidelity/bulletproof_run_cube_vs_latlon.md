# legoESM cube vs lat-lon bulk-metric comparison

Tolerance: relative delta <= 5.0%.

| case | metric | lat-lon | cube | abs Δ | rel Δ | status |
|------|--------|---------|------|-------|-------|--------|
| rest_state_stratified_with_land | T_min | 2.12421 | 2.12421 | 0 | 0.00% | PASS |
| rest_state_stratified_with_land | T_max | 19.5345 | 19.5346 | 2.67e-05 | 0.00% | PASS |
| rest_state_stratified_with_land | T_mean | 5.23022 | 5.23022 | 2.53e-07 | 0.00% | PASS |
| rest_state_stratified_with_land | u_mean | 0 | 2.34996e-19 | 2.35e-19 | 0.00% | PASS |
| rest_state_stratified_with_land | v_mean | 0 | 4.10026e-19 | 4.1e-19 | 0.00% | PASS |
| rest_state_stratified_with_land | ke_mean | 0 | 6.47772e-34 | 6.48e-34 | 0.00% | PASS |
| rest_state_stratified_with_land | u_abs_max | 0 | 3.58129e-17 | 3.58e-17 | 0.00% | PASS |
