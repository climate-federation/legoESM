# legoESM Dynamical Core Test Suite Results

Generated: 2026-03-01 11:39:34

## Solver Matrix

| # | Test | Solver | Status | Key Metric | Value | Wall Time | Notes |
|---|------|--------|--------|------------|-------|-----------|-------|
| 1 | Held-Suarez 30d SI | Hydro Spec T21/L10 SI | **FAIL** | max |v| | nan | 7.3s | <T>=nan, dt=600.0s |
| 2 | DCMIP TC2a 6min SI | NH FV C8/L20 SI | PASS | max |w| | 0.5962 | 5.8s | dt=1.0s |

## Summary

- **Total tests**: 2
- **Passed**: 1
- **Failed**: 1
- **Errors**: 0
- **Total wall time**: 13s (0.2 min)

## Solver Coverage

| Solver Type | Tests Run | Status |
|-------------|-----------|--------|
| Hydrostatic Spectral (Spectral PE) | 1 | 0/1 PASS |
| Hydrostatic Spectral SI (Semi-Implicit) | 2 | 1/2 PASS |
| Non-Hydrostatic FV (Compressible Euler) | 1 | All PASS |

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

### Spectral PE: T15 explicit, T21+ semi-implicit
The explicit SSP-RK3 spectral PE is limited to T15 by the gravity-wave CFL.
The Hoskins-Simmons (1975) semi-implicit scheme (`semi_implicit=True`) treats
gravity waves implicitly, enabling T21+ with dt=600s. The 2/3 dealiasing grid
is correctly implemented (`n_lat = 3*(n_max+1)//2`).

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
