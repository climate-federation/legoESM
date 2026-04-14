# FV3 Fortran Fidelity Review (2026-04-14)

Scope:
- Python side: `src/legoesm/grids/cubed_sphere_cdgrid.py`, `src/legoesm/core/fv3_sw_core.py`, `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py`
- Fortran reference: `../atmos_cubed_sphere-symmetryclean/model/fv_grid_utils.F90`, `../atmos_cubed_sphere-symmetryclean/model/sw_core.F90`

Bottom line:
- The repo contains useful FV3-inspired pieces, but it is **not currently a faithful port** of the original Fortran `c_sw/d2a2c_vect/d_sw` path.
- The largest fidelity gaps are in metric construction, transport/contravariant factors, boundary handling in `c_sw`, and the overall time-stepping architecture.

## Findings

### 1. `cosa_u/cosa_v/rsin_u/rsin_v` are not built the way FV3 builds them

Python:
- `src/legoesm/grids/cubed_sphere_cdgrid.py:666-678`

Current code:
- derives `cosa_u/cosa_v` from `cosa_corner`
- derives `sina_u/sina_v` from `sqrt(1 - cosa^2)`
- stores `rsin_u/rsin_v = 1/sin`

Fortran:
- `../atmos_cubed_sphere-symmetryclean/model/fv_grid_utils.F90:505-518`

FV3 does:
- `cosa_u = 0.5*(cos_sg(i-1,j,3)+cos_sg(i,j,1))`
- `sina_u = 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))`
- `rsin_u = 1/sina_u^2`
- same pattern for `v`

Measured on C16 interior faces:
- `cosa_u/cosa_v`: max abs diff `5.13e-4`
- `rsin_u/rsin_v`: max abs diff `1.17e-1`, max rel diff `9.55e-2`

Correction:
- Build `cosa_u/cosa_v/sina_u/sina_v/rsin_u/rsin_v` from `sin_sg/cos_sg`, not from `cosa_corner`.
- Use FV3’s `1/sin^2` interior factor and preserve the special edge handling from `fv_grid_utils.F90`.

### 2. Active `area_corner/dxc/dyc` path is not the supergrid-style FV3 metric path

Python:
- `src/legoesm/grids/cubed_sphere_cdgrid.py:596-603`
- `src/legoesm/grids/cubed_sphere_cdgrid.py:884-906`

Current code:
- `area_corner` comes from a 4-cell average of haloed `base.area`
- `dxc/dyc` come from haloed cell-center positions

But the same file already has a more FV3-like helper:
- `_compute_supergrid_metrics(...)` at `src/legoesm/grids/cubed_sphere_cdgrid.py:260-390`

Fortran:
- `fv_grid_utils.F90` uses precomputed `area_c_64/dxc_64/dyc_64` as core grid metrics (`fv_grid_utils.F90:135-141`)

Measured on C16 versus `_compute_supergrid_metrics(...)`:
- `area_corner`: mean rel diff `2.57e-1`, max rel diff `3.01e+0`
- `dxc/dyc`: mean rel diff `1.19e-1`, max rel diff `1.03e+0`

Correction:
- Replace the halo-averaged `area_corner/dxc/dyc` construction with the supergrid-based path, then apply explicit edge/vertex synchronization only if needed.
- If cross-face consistency fixes are still required, add them after the FV3 metric construction rather than replacing it.

### 3. `_d2a2c_vect_duogrid` is explicitly an adaptation to a physical convention, not a faithful port

Python:
- `src/legoesm/core/fv3_sw_core.py:68-163`

The code path:
- converts through geographic east/north
- states it is “adapted for the physical convention”
- uses `1/sin`, not FV3’s `rsin_u/rsin_v`

Fortran:
- Duo-grid enters the same `d2a2c_vect` machinery through `gridstruct%dg%is_initialized`, still using FV3’s native metric conventions (`sw_core.F90:3361-3695`)

Correction:
- If the goal is fidelity, do not describe `_d2a2c_vect_duogrid` as FV3-faithful.
- Port the Fortran duogrid branch directly instead of routing through the repo’s physical/geographic convention.

### 4. `_c_sw` is missing FV3’s boundary KE/vorticity special cases

Python:
- `src/legoesm/core/fv3_sw_core.py:418-431`

Current code:
- picks `uc_left/uc_right` and `vc_bot/vc_top` uniformly for KE upwinding

Fortran:
- `../atmos_cubed_sphere-symmetryclean/model/sw_core.F90:325-364`

FV3 special-cases boundary KE/vorticity using `sin_sg/cos_sg` combinations like:
- `uc*sin_sg + v*cos_sg`
- `vc*sin_sg + u*cos_sg`

Correction:
- Port the boundary KE and vorticity branches exactly instead of using the same interior upwind rule at face edges.

### 5. The repo’s production and forward-backward paths are not FV3-equivalent

Production model:
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:372-385`
- It already says: “not a faithful FV3 port”

Forward-backward path:
- `src/legoesm/core/fv3_sw_core.py:629-717`

Current `_d_sw`:
- uses `cgrid_mass_flux_divergence` + Arakawa-Lamb gradient
- does not port `d_sw1`…`d_sw6`

Fortran:
- `sw_core.F90` couples `c_sw` with the native `d_sw*` operators, transport, and damping stack

Correction:
- Stop treating `fv3_forward_backward_step` as an FV3 port; it is a hybrid experiment.
- If fidelity is the target, port the native `d_sw` operator chain as a unit.
- If stability/quality is the target, keep the research path but label it clearly as non-FV3.

### 6. Some local tests currently lock in the non-FV3 behavior

Python test:
- `tests/unit/test_cdgrid_fv3_regression.py:199-227`

These tests assert:
- `rsin_u = 1/sqrt(1-cosa_u^2)`
- `rsin_v = 1/sqrt(1-cosa_v^2)`

That matches the current Python approximation, but not the Fortran formulas in `fv_grid_utils.F90:507-518`.

Correction:
- Update or replace these tests with Fortran-formula regression tests before changing the implementation, otherwise the fidelity fix will look like a regression.

## Recommended order for Claude

1. Fix the metric layer first in `cubed_sphere_cdgrid.py`.
2. Update the tests so they validate the Fortran formulas, not the current approximations.
3. Re-port `_d2a2c_vect` and `_d2a2c_vect_duogrid` against the corrected metrics.
4. Port the missing boundary logic in `_c_sw`.
5. Either:
   - fully port `d_sw`, or
   - clearly separate the research/stabilized path from the faithful FV3 path in names and docs.

## Iteration 1 — Resolved (2026-04-14)

### 1. cosa_u/cosa_v/rsin_u/rsin_v built from sin_sg/cos_sg ✅
- `cosa_u/sina_u` now computed from `0.5*(cos_sg[i-1,j,E] + cos_sg[i,j,W])` matching `fv_grid_utils.F90:505-518`
- `rsin_u` = `1/sin²` for interior, `1/sin` at panel edges (matching `fv_grid_utils.F90:509-554`)
- Verified: interior max rel diff from expected formula = 1.95e-07 (float precision)

### 6. Regression tests updated ✅
- Tests now validate FV3 `1/sin²` interior + `1/sin` edge convention, not the old `1/sqrt(1-cos²)` approximation

### Operator call sites updated ✅
- All 8 locations in `fv3_sw_core.py` that reconstructed `sina = sqrt(1-cos²)` and used `/sin` now use stored `cdgrid.rsin_u` (`1/sin²`) directly
- `_d2a2c_vect`, `_d2a2c_vect_duogrid`, `_c_sw` vorticity flux, `fv3_csw_tendencies` vorticity flux all updated
- `_fv3_forward_backward_step` correctly uses physical velocity (1/sin) since it feeds `cgrid_mass_flux_divergence` which expects physical velocity

### Cell-centre metrics from sin_sg ✅
- `cosa_cell = cos_sg[:,:,:,4]` (cell centre), `sina_cell = sin_sg[:,:,:,4]`
- `rsin2_cell = 1/sin²` — matches FV3 `cosa_s/rsin2` convention

## Remaining fidelity issues

### 2. area_corner/dxc/dyc still non-supergrid approximation
Python: halo-averaged cell areas and centre-to-centre distances.
Fortran: precomputed `area_c_64/dxc_64/dyc_64` from supergrid.
Measured on C16: `area_corner` mean rel diff 2.57e-1, max 3.01e+0; `dxc/dyc` mean 1.19e-1, max 1.03e+0.
Correction: use `_compute_supergrid_metrics()` (currently dead code).

### 3. `_d2a2c_vect_duogrid` is an adaptation, not a faithful port
Routes through geographic east/north convention rather than FV3's native metric conventions.
Correction: port the Fortran duogrid branch of `d2a2c_vect` directly.

### 4. `_c_sw` missing boundary KE/vorticity special cases
Python uses uniform upwind rule at face edges. FV3 uses `uc*sin_sg + v*cos_sg` and `vc*sin_sg + u*cos_sg` at face boundaries (`sw_core.F90:325-364`).

### 5. Production and forward-backward paths not FV3-equivalent
`_fv3_forward_backward_step` is a hybrid experiment (not FV3). `_d_sw_native` partially ports `d_sw` but not the full `d_sw1`…`d_sw6` chain.

### 7. Python lacks haloed sin_sg/cos_sg transport metrics
FV3 fills ghosts and patches tile-corner `sin_sg/cos_sg` entries in `fv_grid_utils.F90:570`. Python uses `mode='edge'` padding at face boundaries, missing the cross-face upwind selection that FV3 provides.

### 8. Boundary ut/vt override drops cross-velocity term
`_d2a2c_vect` boundary override `ut = uc / sin_sg_upwind` omits the `v*cosa` cross-velocity term that FV3 retains: `ut = (uc - v*cosa)*rsin_u` even at boundaries.

## Sanity checks already run

Commands:
- `JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/unit/test_cdgrid_fv3_regression.py tests/unit/test_duogrid.py` → 86 passed
- `JAX_ENABLE_X64=1 scripts/run_atmosphere_test_matrix.py --only sw --grid cubed_sphere --quick` → 3 PASS
  - Williamson 2: L2=1.94e-03, Linf=8.66e-03
  - Williamson 5: mass drift=1.56e-05
  - Cosine bell: L1=1.41e-01, L2=1.35e-01, Linf=1.49e-01
- Ocean rest state: all 4 cubed-sphere variants PASS (eta drift 1e-14 to 1e-18)
- No visible edge artifacts in v-wind, wind speed, or cosine bell snapshots
