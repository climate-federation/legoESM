# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State after iter-1..9 (compressed at iter-10)

**Wins**:
- Cube SW W2/W5: matrix runner now uses iter-1030 calibration (`damp_v=0.030`) via the canonical `iter1009_dual_target_config(n)` helper (iter-1/8). v_ll_Linf 5-day = 3.65 m/s (PASS).
- Cube NH: parity gap CLOSED for ALL THREE DCMIP cases via `use_fv3_vector_halo_uv=True` + `use_fv3_a2b_ord4_vector_uv=True` (iter-697/698 flags promoted to matrix runner, iter-5/6/7):
  - TC1: 0.3266 → **0.0142 m/s** (23x), matches ico 0.0145
  - TC2: 4.6528 → **0.3177 m/s** (14.6x), matches ico 0.3568
  - TC3: 23.07 → **7.36 m/s** (3.1x), now BEATS ico 10.24
- Cube transport flux closure verified bit-clean in fp64 (iter-3); `transport_step` flux differencing promoted to fp64 (`core/fv_tp_2d.py:1117-1131`).

**Regression sentinels added** (iter-4 + iter-9 + iter-10):
- `tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py::TestCubeCswW2Residual` — pins cube c_sw + p_grad_c residual on W2 IC.
- `tests/test_matrix_nh_cube_parity_ast_guard.py` — 6 AST tests pin iter-697/698 flags in all 3 NH cube branches (catches removal in 0.10 s).
- iter-329 regex updated to accept 4-arg `center_to_dgrid_vector(u, v, cdgrid, use_fv3_a2b_ord4=...)`.

## Structural cube parity findings (not closed)

- **c_sw + p_grad_c cube-vertex residual** (codex iter-983; iter-3/4 reproduced): |duc|_max = 2.71 m/s at face=2 i=1 j=35 on cube W2 IC. Decomposition: fy1·vort_x = -2.98 m/s (dominant), dke_x = +1.32, dp_x = +1.16, combined = -0.50 m/s. Interior point gives 2e-6 residual — 250000× ratio. STRUCTURAL to duogrid c_sw path; FV3 oracle also skips corner correction on duogrid (`sw_core.F90:395-401`). Codex 1000+ iters reached v_ll=0.114 via calibration damping, not by closing the gap.
- **Cube CB no-anchor mass drift** (iter-2): raw `transport_step` (mass_target=None) at C36 drifts 3.82e-4/day vs latlon raw 1.49e-5. Exponential blow-up step 40+ from PPM monotone limiter producing small negative h cells at cube vertices. Anchor's positivity clip (`jnp.maximum(h_new, 0.0)`) is the correct fix; flux closure itself is bit-clean.
- **PE cube already best-in-class** (iter-9 probe): matrix-runner survey shows cube wins or ties on baroclinic / rotated_steady / held_suarez / amip max|v|. `use_fv3_a2b_zeta_corner=True` (PE analog of NH iter-697/698) was neutral on rotated_steady (+30% wall, same numerics); reverted.

## Iter trail (terse)

- iter-1: matrix W2/W5 cube `damp_v` 0.06→0.030 (iter-1030 sentinel-pinned). W2 5-day PASS.
- iter-2: cube CB no-anchor probe — 3.82e-4 drift, blow-up step 40+. PPM positivity structural. Revert.
- iter-3: cube flux closure bit-clean fp64; promoted `transport_step` differencing to fp64; c_sw probe found 2.71 m/s residual at face=2 i=1 j=35; hord sweep all blow up similarly.
- iter-4: regression sentinel `TestCubeCswW2Residual` pins `|duc|_max < 3.0, |dvc|_max < 3.6`.
- iter-5: NH TC1 cube parity CLOSED (23×) — added `use_fv3_vector_halo_uv` + `use_fv3_a2b_ord4_vector_uv`.
- iter-6: NH TC2 cube parity CLOSED (14.6×) — same flags.
- iter-7: NH TC3 cube parity CLOSED (3.1×) — same flags. All 3 NH DCMIP cube-parity-clean.
- iter-8: DRY refactor — matrix W2/W5 uses canonical `iter1009_dual_target_config(n)` helper.
- iter-9: PE flag (`use_fv3_a2b_zeta_corner`) probe neutral, reverted; AST guard sentinel for iter-5/6/7 NH flags.
- iter-10 (compressed at this point): fix outdated iter-329 regex (accept 4-arg `center_to_dgrid_vector`); compress this doc; queue iter-11+ targets.
- iter-11: extend AST guard sentinel to pin the iter-1/iter-8 SW helper substitution (`iter1009_dual_target_config(n)` in matrix runner).  Now 7/7 AST tests covering both NH cube flags (iter-5/6/7) and SW cube helper (iter-1/8) in 0.09 s.  Sanity-checked the 14 AST + a2b_ord4 wiring sentinels — all PASS.
- iter-12: enable `use_fv3_d_con_cv=True` on all 3 NH cube branches (TC1/TC2/TC3) — FV3 iter-320 correctness fix: NH conserves internal energy c_v·T but legoESM default used c_pd, under-heating d_con by c_v/c_p ≈ 0.714 (~40 % magnitude).  Factory ``make_fv3_faithful_nh_config`` enables this by default; matrix had remained at the under-heating default.  TC1 |w|_max verified identical at 0.0142 m/s (the d_con flag affects heating, not the |w| diagnostic).  AST guard extended to 8 tests pinning the iter-12 flag in all 3 cube cases.
- iter-13: enable `use_fv3_dynamic_exner=True` + `use_fv3_metric_aware_d_con=True` on all 3 NH cube branches (factory defaults; PE/NH iter-336/339).  iter-336 dynamic Exner uses Π_total = Π_ref + π' (matches FV3 live pkz) at the 3 slow-tendency d_con sites.  iter-339 metric-aware d_con uses the rsin2/cosa_s form at the damp_v d_con site (PE iter-338 mirror).  Both compose with iter-12 d_con_cv.  TC1 cube C36 quick: |w|_max=0.0142 m/s unchanged (these flags affect d_con heating, not the |w| diagnostic).  AST guard extended to 10 tests covering the iter-13 flags in all 3 NH cube branches; 0.12 s.

## Iter-11+ queued

- Probe AMIP cube vs other grids (matrix already shows 4.18e-11 mass drift — best-in-class).
- Investigate if any iter-697/698 / iter-14 helper flag improves NH or PE numerics at FINER resolutions (C48, C72) — current results are at C36.
- Look at SW Williamson 6 (Rossby-Haurwitz) cube wiring — currently only ico/spectral run W6.
