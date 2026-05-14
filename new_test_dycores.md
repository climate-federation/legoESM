# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State after iter-1..19 (compressed at iter-20)

**Major wins**:
- **NH cube parity gap CLOSED** for all 3 DCMIP cases (iter-5/6/7):
  - TC1: 0.3266 → **0.0142 m/s** (23×), matches ico 0.0145
  - TC2: 4.6528 → **0.3177 m/s** (14.6×), matches ico 0.3568
  - TC3: 23.07 → **7.36 m/s** (3.1×), now BEATS ico 10.24
- **PE cube gravity_wave_3_1 parity improved** (iter-18): max|v| 27.5 → **22.4 m/s** (-18.5 %), now within 12 % of ico 20.0 / spec 20.3 / latlon 20.2 cluster (was 38 % outlier).
- **SW cube W2/W5 calibration** adopted iter-1030 sentinel-pinned `damp_v=0.030` (iter-1) + factored to `iter1009_dual_target_config(n)` helper (iter-8).

**NH cube config — full FV3-faithful factory bundle now in matrix runner** (iter-5/6/7/12/13/14/15/16/17):
  - iter-5/6/7: `use_fv3_vector_halo_uv=True`, `use_fv3_a2b_ord4_vector_uv=True` (the dominant parity-closing pair)
  - iter-12: `use_fv3_d_con_cv=True` (c_v denominator, ~40 % heating-magnitude correctness)
  - iter-13: `use_fv3_dynamic_exner=True`, `use_fv3_metric_aware_d_con=True`
  - iter-14: `d_con_top_zero_levels=2`
  - iter-15: `heat_source_del2_iters=2`
  - iter-16: `delt_max=1.0`
  - iter-17: `corner_div_damp_nord=1`, `corner_div_damp_d4_bg=0.16`

**PE cube config — partial factory bundle** (iter-18/19):
  - iter-18: `use_fv3_metric_aware_d_con=True` (the active parity-improving flag)
  - iter-19: `d_con_top_zero_levels=2`
  - Skipped: `use_fv3_a2b_zeta_corner=True` (51 % wall cost, neutral on rotated_steady + gravity_wave_3_1 — iter-9 + iter-19 probes)

**Code hardening**:
- `core/fv_tp_2d.py:transport_step` flux differencing promoted to fp64 (iter-3); consistent with the iter-1..64 mainline fp64 budget pattern.

**Regression sentinels**:
- `tests/test_matrix_nh_cube_parity_ast_guard.py` — 16 AST tests pinning iter-5/6/7/8 + iter-12..19 flag wiring; 0.17 s.
- `tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py::TestCubeCswW2Residual` — pins cube c_sw + p_grad_c residual on W2 IC (iter-4).
- `tests/test_nh_vector_halo_ast_guard_iter329.py` — pre-existing failure fixed in iter-10 (regex now accepts 4-arg ``center_to_dgrid_vector``).

## Structural cube parity findings (not closed)

- **c_sw + p_grad_c cube-vertex residual** (codex iter-983; iter-3/4): |duc|_max = 2.71 m/s at face=2 i=1 j=35 on cube W2 IC. STRUCTURAL to duogrid c_sw path; FV3 oracle also skips corner correction on duogrid. Codex 1000+ iters reached v_ll=0.114 via calibration damping, not by closing the gap.
- **Cube CB no-anchor mass drift** (iter-2): raw `transport_step` drifts 3.82e-4/day vs latlon raw 1.49e-5 from PPM monotone limiter producing small negative h cells at cube vertices. Anchor's positivity clip is the correct fix.
- **TC3 cube cumulative iter-12..17 verification was in flight at iter-18** but completed after iter-19 commit; not formally re-measured. iter-7's TC3 result was 7.36 m/s; subsequent flags are correctness fixes that don't affect |w| diagnostic on TC1/TC2, so TC3 expected to remain near 7.36.

## Iter trail (terse)

- iter-1: SW W2/W5 cube `damp_v` 0.06→0.030 (iter-1030 calibration); W2 5-day PASS.
- iter-2: cube CB no-anchor probe — 3.82e-4 drift, PPM positivity structural.
- iter-3: cube flux closure bit-clean fp64; promoted `transport_step` differencing to fp64; c_sw probe found 2.71 m/s residual at face=2 i=1 j=35.
- iter-4: regression sentinel `TestCubeCswW2Residual` pins `|duc|_max < 3.0, |dvc|_max < 3.6`.
- iter-5: NH TC1 cube parity CLOSED (23×) — `use_fv3_vector_halo_uv` + `use_fv3_a2b_ord4_vector_uv`.
- iter-6: NH TC2 cube parity CLOSED (14.6×).
- iter-7: NH TC3 cube parity CLOSED (3.1×).
- iter-8: DRY — matrix W2/W5 uses canonical `iter1009_dual_target_config(n)` helper.
- iter-9: PE `use_fv3_a2b_zeta_corner` probe neutral, reverted; AST guard added.
- iter-10: compress doc; fix iter-329 regex (4-arg `center_to_dgrid_vector`).
- iter-11: AST sentinel for iter-1/8 SW helper substitution.
- iter-12: NH `use_fv3_d_con_cv=True` on all 3 NH cube branches.
- iter-13: NH `use_fv3_dynamic_exner` + `use_fv3_metric_aware_d_con`.
- iter-14: NH `d_con_top_zero_levels=2`.
- iter-15: NH `heat_source_del2_iters=2`.
- iter-16: NH `delt_max=1.0`; cumulative iter-12..16 verified TC1 + TC2.
- iter-17: NH `corner_div_damp_nord=1` + `corner_div_damp_d4_bg=0.16`.
- iter-18: PE `use_fv3_metric_aware_d_con=True` — gravity_wave_3_1 max|v| 27.5 → 22.4 (-18.5 %).
- iter-19: PE `d_con_top_zero_levels=2`; reverted `use_fv3_a2b_zeta_corner` (51 % wall, neutral).
- iter-20 (compressed at this point): doc compression + PE `delt_max=1.0` on baroclinic cube (PE iter-218 factory default, analog of NH iter-16); gravity_wave_3_1 cube max|v|=22.4 unchanged, mass_drift=5.01e-12, wall 58.6 s; AST guard now 17 tests.
- iter-21: PE `heat_source_del2_iters=2` on baroclinic cube (PE iter-458 factory default; del-2 smoothing of `_d_con_sum` heat source; analog of NH iter-15).  gravity_wave_3_1 cube max|v|=22.4 / mass_drift=5.01e-12 unchanged; wall 57.8 s.  AST guard now 18 tests.
- iter-22: propagate the iter-18..21 PE factory bundle (`use_fv3_metric_aware_d_con` + `d_con_top_zero_levels=2` + `delt_max=1.0` + `heat_source_del2_iters=2`) to held_suarez cube + amip cube PE configs (was only on baroclinic).  All 3 PE cube branches now match the factory.  Sanity-verified gravity_wave_3_1 (baroclinic branch) still 22.4 m/s unchanged at 57.9 s.  AST guard count-based sentinel covers all 3 PE cube branches; now 19 tests.
- iter-23: **Cumulative-bundle long-run verification of iter-12..22.**  Two independent measurements:
  - **NH TC3 cube cumulative iter-12..17 (8 NH flags total)**: ran C36 quick (4 min, 1080 steps).  |w|_max=**7.3635 m/s, mass_drift=5.16e-16** — bit-identical to iter-7's measurement.  Wall 2436 s (+19 % vs iter-7's 2042 s from the 5 added flags).  All correctness flags compose cleanly on the squall-line + Kessler test.
  - **PE held_suarez cube iter-22 bundle (sigma vertical)**: ran C36 30-day.  max|v|=**10.8 m/s, mass_drift=4.97e-12** — bit-identical to pre-iter-22 measurement.  Wall 1131 s (+1.5 % vs 1116 s pre-change).  The PE factory bundle is also stable on the 30-day climate-equilibrium run.
  
  Both NH cube parity (iter-5/6/7 win at 7.36 m/s on TC3) AND PE cube held_suarez stability are preserved through the iter-12..22 cumulative bundle.
- iter-24: **SW Williamson 6 cube wired (deferred from iter-11).**  Add `test_num=6` branch to the matrix runner SW cube path using `_w6_winds_geo(lon_edge, lat_edge, R)` from `tests/test_cases/williamson_extended.py` for analytic edge-midpoint (u_east, v_north) at every D-grid edge; rotate to (u_d, v_d) via the cube's `(cos_angle_edge, sin_angle_edge)` matrices.  h field from `williamson_test6(grid)` cell-centre init.  Also extended the W6 GRID_TYPES iteration (line 211) to include `cubed_sphere`.  **Result at C36 day-1 quick**: cube `mass_drift=3.82e-16, wall 12.6 s` — machine precision; cross-grid table now shows cube/ico/spec at 3.82e-16 / 0.00e+00 / 1.91e-16.  Lat-lon W6 still deferred (no W6 C-grid wind init yet).
- iter-25: **SW Williamson 6 lat-lon (C-grid) wired (deferred from iter-24).**  Add `test_num==6` inline init to the matrix runner SW latlon path using `_w6_winds_geo` evaluated directly at u-face (lon_face × cell-lat) and v-face (cell-lon × lat_face) coordinates — latlon C-grid faces align with east/north so no rotation needed.  v at the poles is safely 0 (W6 v_north has a `cos(lat)^(R-1)` factor with R=4).  Also imported `CGridLatLonShallowWaterState` (was missing).  Extended the W6 GRID_TYPES iteration to include `"latlon"`.  **Result at 72x144 day-1 quick**: latlon `mass_drift=1.91e-16, wall 6.1 s`.  **All 4 grid types now run W6 at machine-precision mass conservation** (cube 3.82e-16 / ico 0.0 / latlon 1.91e-16 / spec 1.91e-16).

## Iter-21+ queued (status post iter-24)

- ✅ iter-21: PE `heat_source_del2_iters=2` on baroclinic cube.
- ✅ iter-22: PE factory bundle propagated to held_suarez + AMIP cube.
- ✅ iter-23: TC3 cube cumulative iter-12..17 verified (7.36 m/s bit-identical to iter-7).
- ✅ iter-24: SW Williamson 6 cube wiring (mass_drift=3.82e-16, cross-grid table now includes cube W6).
- ✅ iter-25: SW Williamson 6 lat-lon wiring (mass_drift=1.91e-16); all 4 grids on W6 cross-grid table.
- AMIP cube cumulative bundle verification (slow — 30-day quick at 900 s wall).
