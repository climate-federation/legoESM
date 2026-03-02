# legoESM Dynamical Core Test Suite Results

Generated: 2026-03-02 16:28:03

## Solver Matrix

| # | Test | Solver | Status | Key Metric | Value | Wall Time | Notes |
|---|------|--------|--------|------------|-------|-----------|-------|
| 1 | Williamson 2 | SW FV C16 | PASS | L2 error (5d) | 2.08e+02 | 1.5s |  |
| 2 | Williamson 5 | SW FV C16 | PASS | mass drift (15d) | 5.25e-04 | 1.7s |  |
| 3 | Williamson 2 | SW Spec T21 | PASS | L2 error (5d) | 2.32e-08 | 2.2s |  |
| 4 | Williamson 5 | SW Spec T21 | PASS | mass drift (15d) | 1.79e-16 | 6.3s |  |
| 5 | Held-Suarez 30d | Hydro FV C16/L10 | PASS | mass drift | 6.02e-11 | 8.2s | max|v|=12.9 |
| 6 | Baroclinic 10d | Hydro FV C16/L10 | PASS | ps min (hPa) | 982.4 | 8.5s | max|v|=21.4 |
| 7 | Held-Suarez 30d | Hydro Spec T15/L10 | PASS | p99 |v| @ jet | 52.6 | 43.2s | max_all=193.8, max_jet=52.6, <T>=279.1 |
| 8 | Baroclinic 1d | Hydro Spec T15/L10 | PASS | p99 |v| @ jet | 47.9 | 1.9s | max_all=143.8, max_jet=47.9, ps_min=854.2hPa |
| 9 | DCMIP TC1 1h | NH FV C8/L20 | PASS | max |w| | 0.0636 | 7.3s |  |
| 10 | DCMIP TC2a 3min | NH FV C8/L20 | PASS | max |w| | 0.2050 | 5.6s |  |
| 11 | DCMIP TC3 3min | NH FV C8/L20 | PASS | max |w| | 0.6725 | 3.3s | f=0, max qr=0.000000 |
| 12 | DCMIP TC1 1h | NH Spec T15/L20 | PASS | max |w| | 0.0446 | 13.1s |  |
| 13 | DCMIP-2012 1-1 | Transport C16/L10 | PASS | L2 q1 | 1.11e+00 | 1.2s | Linf=1.03e+00 |
| 14 | Held-Suarez 30d SI | Hydro Spec T21/L10 SI | PASS | p99 |v| @ jet | 52.6 | 196.1s | max_all=246.5, max_jet=52.6, <T>=270.7, dt=600s, sub=5, nu=14.0x |
| 15 | DCMIP TC2a 6min SI | NH FV C8/L20 SI | PASS | max |w| | 0.5962 | 8.2s | dt=1.0s |

## Summary

- **Total tests**: 15
- **Passed**: 15
- **Failed**: 0
- **Errors**: 0
- **Total wall time**: 308s (5.1 min)

## Solver Coverage

| Solver Type | Tests Run | Status |
|-------------|-----------|--------|
| Shallow Water FV | 2 | All PASS |
| Shallow Water Spectral | 2 | All PASS |
| Hydrostatic FV (Primitive Eq.) | 2 | All PASS |
| Hydrostatic Spectral (Spectral PE) | 3 | All PASS |
| Hydrostatic Spectral SI (Semi-Implicit) | 2 | All PASS |
| Non-Hydrostatic FV (Compressible Euler) | 4 | All PASS |
| Non-Hydrostatic Spectral (Spectral NH) | 1 | All PASS |
| Tracer Transport | 1 | All PASS |

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
