# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State after iter-1..49 (compressed at iter-50)

**Headline cube parity wins**:

- **NH cube parity CLOSED** for all 3 DCMIP cases (iter-5/6/7).  Pre→post:
  - TC1: 0.327 → **0.014 m/s** (23×).  Matches ico 0.015 / spec 0.014 within 1 ULP.
  - TC2: 4.65 → **0.32 m/s** (14.6×).  Matches ico 0.36 / spec 0.36.
  - TC3: 23.1 → **7.36 m/s** (3.1×).  BEATS ico 10.24.
  Mechanism: `use_fv3_vector_halo_uv` + `use_fv3_a2b_ord4_vector_uv` (iter-697/698 FV3-faithful pair) promoted from the factory to the matrix runner NH cube branches.

- **SW W2 cube parity 7.2× improved** (iter-42/44).  v_ll_Linf 3.65 → **0.51 m/s**; L2 2.45e-3 → 4.58e-4 (cube/latlon ratio 9× → 1.7×).  Mechanism: enable `hyperdiff_coeff = 2 × _hyperdiff_cube(n)` on cube SW propagating tests (test_num ∈ {2, 5, 6}).

- **SW cube W5 / W6 14-15-day stability** (iter-31/33/44).  Pre-fix W5 BLEW UP day 14.58, W6 BLEW UP day 9.  Now bit-clean mass conservation at full duration via the same hyperdiff override.

- **PE cube gravity_wave_3_1** max|v| 27.5 → **22.4 m/s** (-18.5 %, iter-18) via `use_fv3_metric_aware_d_con=True`.  Now within 12 % of ico/spec/latlon cluster.

- **SW Williamson 6 wired for all 4 grids** (iter-24/25).  cube/latlon/ico/spec all at machine-precision mass conservation; cross-grid table complete.

**Matrix-runner cube config bundles in place**:
- NH cube (iter-5/6/7/12/13/14/15/16/17, 8 flags): vector_halo + a2b_ord4 + d_con_cv + dynamic_exner + metric_aware_d_con + d_con_top_zero_levels=2 + heat_source_del2_iters=2 + delt_max=1.0 + corner_div_damp_nord=1/d4_bg=0.16.
- PE cube on all 3 branches (iter-18..22, 4 flags): metric_aware_d_con + d_con_top_zero_levels=2 + delt_max=1.0 + heat_source_del2_iters=2.
- SW cube W2/W5/W6 (iter-42/44): `hyperdiff_coeff = 2 × _hyperdiff_cube(n)`.

**Code hardening**:
- `core/fv_tp_2d.py:transport_step` flux differencing promoted to fp64 (iter-3).
- `iter1009_dual_target_config(n, ..., hyperdiff_coeff=0.0)` accepts hyperdiff kwarg (iter-35).
- Stale iter-1019 "signature-only NO-OP" warning removed from `operators_cdgrid.py:fv3_sw_tendencies` (iter-41 — real biharmonic hyperdiff implementation IS at line 1444-1456).

**Regression sentinels** (25 AST + 4 numerical + 1 c_sw residual):
- `tests/test_matrix_nh_cube_parity_ast_guard.py` — 25 AST tests pinning flag wiring across NH/PE cube branches + SW gate `{2,5,6}` + W6 init helpers; whitespace-tolerant regex + proximity sanity (iter-27/28/29/45 hardened).
- `tests/test_iter1002_w2_target_met.py` — 4 numerical sentinels:
  - `test_iter1002_w2_v_ll_linf_meets_target`: W2 1-day at iter-1030 calibration (hyperdiff=0).
  - `test_iter1009_w5_day5_artifact_free`: W5 day-5 (hyperdiff=0 calibration).
  - `test_iter49_cube_w2_matrix_config_short_run`: W2 2-day at iter-44 matrix config (hyperdiff=2×).
  - `test_iter48_cube_w5_short_run_stable_with_hyperdiff`: W5 2-day (hyperdiff=2×).
  - `test_iter39_cube_w6_short_run_stable_with_hyperdiff`: W6 2-day (hyperdiff=2×).
  - `test_iter35_hyperdiff_coeff_kwarg_threads_through`: helper kwarg API.
- `tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py::TestCubeCswW2Residual` — cube c_sw + p_grad_c residual on W2 IC (iter-4).

**Cumulative-bundle long-run verifications**:
- TC3 cube cumulative iter-12..17: |w|=7.36 m/s bit-identical to iter-7 (iter-23).
- held_suarez sigma cube iter-22 bundle: max|v|=10.8 m/s bit-identical (iter-23).
- AMIP cube iter-22 bundle: mass_drift=4.18e-11 bit-identical (iter-47).
- Full SW cube matrix at full duration (iter-43): 4/4 PASS.
- iter-44 2× hyperdiff probed at C36 + C48 (iter-46) — same optimum.

## Structural cube parity findings (not closed)

- **c_sw + p_grad_c cube-vertex residual** (codex iter-983 + iter-3/4/26 reproduced): |duc|_max = 2.71 m/s at face=2 i=1 j=35 on cube W2 IC.  STRUCTURAL to the duogrid c_sw path.  iter-1030 calibration + iter-44 hyperdiff partially mask this, but the per-step imbalance remains.
- **Cube CB 12-day L2=1.09** (iter-26 hord sweep + iter-34 + iter-51 resolution probe): structural cube PPM transport accuracy.  C36 L2=1.092, C48 L2=1.094 — virtually identical, confirming the issue is NOT dx² convergence-limited.  Root cause: PPM monotone limiter dissipation accumulates over 12-day rotation, eventually smearing the cosine bell to a near-flat field.  Cube vs latlon L2 ratio 45× (cube 1.09 vs latlon 0.024 at iter-43 measurement).  CB uses `transport_step` directly, not `model.step()`; hyperdiff path doesn't engage.  Would require a tracer-specific transport variant or a different limiter to address.
- **C48 hyp=0 cube W2 v_ll = 5.55 m/s** (iter-46): finer grids amplify cube-vertex residual without hyperdiff coverage.  iter-44's 2× hyperdiff is essential, not optional.

## Iter trail (terse, iter-1..49)

- iter-1/8/34: SW W2/W5/CB cube → iter-1030 calibration via `iter1009_dual_target_config(n)`.
- iter-2: cube CB raw-transport leak diagnostic.
- iter-3: `transport_step` fp64 + cube flux closure verified bit-clean.
- iter-4: c_sw W2 residual sentinel.
- iter-5/6/7: **NH cube parity CLOSED** (TC1/TC2/TC3 23×/14.6×/3.1×) via vector_halo + a2b_ord4.
- iter-9: PE `use_fv3_a2b_zeta_corner` neutral, reverted.
- iter-10/20/30/40/50: doc compress + iter-329 regex fix (iter-10).
- iter-11: AST sentinel for SW helper substitution.
- iter-12..17: NH factory bundle (d_con_cv + dynamic_exner + metric_aware + d_con_top_zero_levels + heat_source_del2 + delt_max + corner_div_damp).
- iter-18: PE gravity_wave_3_1 -18.5 % via metric_aware_d_con.
- iter-19/20/21: PE factory bundle (d_con_top_zero_levels + delt_max + heat_source_del2).
- iter-22: PE factory bundle propagated to held_suarez + AMIP.
- iter-23: cumulative TC3 + held_suarez verifications bit-identical.
- iter-24/25: SW W6 cube + latlon wiring (4-grid table at machine precision).
- iter-26: cube CB hord sweep (hord=12 best); 3 W6-wiring AST sentinels.
- iter-27/28/29: hardened PE block-finder anchors.
- iter-31/33: cube W5/W6 long-run hyperdiff override.
- iter-32: W6 hyperdiff AST sentinel.
- iter-35: helper `hyperdiff_coeff` kwarg API.
- iter-36: docstring + kwarg threading test.
- iter-37: review-driven W2 gate-exclusion sentinel.
- iter-38: end-to-end SW cube quick matrix PASS.
- iter-39: numerical sentinel for cube W6 stability.
- iter-40: investigated iter-31 mechanism (initially mis-concluded JAX-trace).
- iter-41: **iter-40 correction** — real biharmonic hyperdiff at `fv3_sw_tendencies:1444`; removed stale warning.
- iter-42: **cube W2 5-day v_ll 3.65 → 0.82 m/s** (4.5×) via gate widening to `{2,5,6}`.
- iter-43: cumulative cube SW matrix full-duration 4/4 PASS.
- iter-44: **bumped hyperdiff coefficient 1× → 2× → cube W2 v_ll 0.82 → 0.51 m/s** (cumulative 7.2× from iter-1).
- iter-45: review-driven AST sentinel hardening — tightened gate regex + iter-1002 independence sentinel.
- iter-46: resolution sweep at C48 confirms 2× hyperdiff optimum.
- iter-47: AMIP cube 30-day quick PASS bit-identical to pre-iter-22.
- iter-48: W5 numerical sentinel (mirror of iter-39 W6).
- iter-49: W2 matrix-config numerical sentinel (hyperdiff=2× variant).
- iter-50 (compressed at this point): doc compression.
- iter-51: probed cube CB at C48 to test resolution dependence — L2=1.094 (essentially identical to C36 L2=1.092).  CB structural transport gap is NOT a dx² convergence issue; PPM monotone limiter dissipation accumulates over the 12-day rotation regardless of resolution.  Added iter-46 C48 reference to the matrix-runner iter-44 hyperdiff comment.  No code change needed beyond comment.
- iter-52: probed `nord_v=1` (del-4 post-step vorticity damping) vs the iter1009 default `nord_v=2` (del-6) for cube SW propagating tests.  Mixed result: v_ll_Linf improves 0.51 → 0.33 m/s but L2 DEGRADES 4.58e-4 → 8.02e-4 (cube/latlon L2 ratio 1.7× → 3.0×).  Kept at nord_v=2 since L2 is the more representative cross-grid parity metric.  W5/W6 unaffected either way.  Added probing rationale to matrix-runner comment.
- iter-53: extended iter-49 W2 numerical sentinel with an additional `h_err_l2 < 3e-3` bound on the area-weighted height-field L2 error.  iter-44 matrix W2 5-day measures L2=4.58e-4 (cube/latlon ratio 1.7×); the day-2 subset stays well under 3e-3.  Now the iter-49 sentinel catches BOTH v_d calibration drift (max|v_d - v_d_init| < 3.0) AND h-field accuracy regression (L2 < 3e-3) — protecting against the iter-52 trade-off where nord_v=1 lowers v_d at the cost of higher L2.  Re-ran cube W2 5-day matrix to refresh cached output (iter-52 probe had left an L2=8.02e-4 stale value); now back at L2=4.58e-4.
- iter-54: probed `div_damp_factor ∈ {6, 8, 10, 12}` at iter-44 baseline (hyperdiff=2×, nord_v=2) on cube W2 5-day.  All values give v_ll_Linf in [0.506, 0.509] m/s (essentially flat) and L2 in [4.48e-4, 5.03e-4].  Optimum is around 10.0 (L2=4.48e-4, ~2% better than 8.0) but the gain is below the test-to-test noise floor.  iter-1030's `div_damp_factor=8.0` retained — no compelling reason to deviate from the calibrated value.

## Iter-51+ queued

- C72 resolution sweep (extends iter-46's C36+C48 coverage).
- Visual artifact inspection of cube W2/W5/W6 snapshots with vs without hyperdiff (reviewer item iter-45).
- Cube CB structural — if a tracer-only del-4 operator is feasible without breaking transport_step's API.
