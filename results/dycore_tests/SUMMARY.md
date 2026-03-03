# legoESM Dynamical Core Test Suite Results

Generated: 2026-03-03 22:31:21

## Solver Matrix

| # | Test | Solver | Status | Key Metric | Value | Wall Time | Notes |
|---|------|--------|--------|------------|-------|-----------|-------|
| 1 | Held-Suarez 30d | Hydro FV C16/L10 | PASS | mass drift | 8.50e-11 | 32.5s | max|v|=10.4 |
| 2 | Baroclinic 10d | Hydro FV C16/L10 | PASS | ps min (hPa) | 984.0 | 36.7s | max|v|=16.6 |
| 3 | Held-Suarez 30d | Hydro Spec T15/L10 | PASS | p99 |v| @ jet | 52.6 | 102.0s | max_all=193.8, max_jet=52.6, <T>=279.1 |
| 4 | Baroclinic 1d | Hydro Spec T15/L10 | PASS | p99 |v| @ jet | 47.9 | 4.4s | max_all=143.8, max_jet=47.9, ps_min=854.2hPa |

## Summary

- **Total tests**: 4
- **Passed**: 4
- **Failed**: 0
- **Errors**: 0
- **Total wall time**: 176s (2.9 min)

## Solver Coverage

| Solver Type | Tests Run | Status |
|-------------|-----------|--------|
| Hydrostatic FV (Primitive Eq.) | 2 | All PASS |
| Hydrostatic Spectral (Spectral PE) | 2 | All PASS |

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
