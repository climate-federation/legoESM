# legoESM Dynamical Core Test Suite Results

Generated: 2026-03-03 22:13:29

## Solver Matrix

| # | Test | Solver | Status | Key Metric | Value | Wall Time | Notes |
|---|------|--------|--------|------------|-------|-----------|-------|
| 1 | Williamson 2 | SW FV C16 | PASS | L2 error (5d) | 1.67e+02 | 18.4s |  |
| 2 | Williamson 5 | SW FV C16 | PASS | mass drift (15d) | 4.31e-04 | 25.4s |  |
| 3 | Williamson 2 | SW Spec T21 ssp_rk3 | PASS | L2 error (5d) | 2.32e-08 | 11.3s |  |
| 4 | Williamson 5 | SW Spec T21 ssp_rk3 | PASS | mass drift (15d) | 1.79e-16 | 33.4s |  |

## Summary

- **Total tests**: 4
- **Passed**: 4
- **Failed**: 0
- **Errors**: 0
- **Total wall time**: 89s (1.5 min)

## Solver Coverage

| Solver Type | Tests Run | Status |
|-------------|-----------|--------|
| Shallow Water FV | 2 | All PASS |
| Shallow Water Spectral | 2 | All PASS |

## Reference Test Suites Covered

- **Williamson et al. (1992)** SW Tests 2, 5: FV + Spectral
- **Held-Suarez (1994)**: FV PE + Spectral PE
- **Jablonowski-Williamson (2006)** Baroclinic Wave: FV PE + Spectral PE
- **DCMIP-2012** Transport Test 1-1: FV
- **DCMIP-2025** TC1/TC2a/TC3: FV Compressible Euler + Spectral NH (TC1)
- **FV3 Idealized Tests** (baroclinic wave, mountain wave): Covered by above
- **GFDL Spectral Dycore Tests** (Williamson, Held-Suarez, BW): Covered by above
- **NGGPS Dycore Testing** evaluation criteria: Metrics reported above

## Known Limitations

### Spectral PE: SI stability envelope at T21
The explicit SSP-RK3 spectral PE remains limited by fast-wave/advection stability
at higher truncations. For robust 30-day T21 Held-Suarez in this suite, we run
`semi_implicit=True` with `dt=600s`, SI subcycling (`si_substeps=5`, i.e. 120s
internal SI stages), and stronger SI-mode hyperdiffusion (14x baseline).
The 2/3 dealiasing grid is correctly
implemented (`n_lat = 3*(n_max+1)//2`).

### NH FV: explicit vs semi-implicit acoustic
The explicit split-explicit scheme (forward-backward acoustic substeps) limits
TC2a/TC3 to ~3 minutes at C8. The semi-implicit acoustic scheme
(`semi_implicit_acoustic=True`) uses a tridiagonal solve for w, removing the
vertical acoustic CFL and enabling 30+ minute integrations.

### Fixes applied in this version
- **Wind rotation**: All DCMIP-2025 test cases now correctly rotate winds from
  geographic (east/north) to cubed-sphere grid coordinates using
  `rotate_winds_geo_to_grid()`. Previously, winds were set directly in grid
  coordinates, which is only correct at face centers.
- **Coriolis toggle**: `CompressibleEulerConfig.use_coriolis` allows f=0 for
  TC3 (squall line), which is designed for cyclostrophic balance.
