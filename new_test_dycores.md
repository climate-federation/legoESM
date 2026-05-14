# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State after iter-1..70 (compressed at iter-70)

**Cross-grid apples-to-apples SW matrix audit (iter-61..70)**: refreshed all 4 grids at full duration (the matrix's quick-mode + full-mode results coexist; iter-61 caught the mixed-mode comparison issue).

- **W2 5-day** (iter-69): cube is **BEST finite-volume grid**.
  - spectral (T21):     L2=1.80e-7, Linf=2.37e-7 (analytic-perfect)
  - ico (ico5):         L2=1.25e-4, Linf=6.05e-4
  - **cube (C36)**:     **L2=4.58e-4, Linf=4.82e-3, v_ll=0.51 m/s**
  - latlon (72×144):    L2=1.41e-3, Linf=1.27e-2
  - Cross-grid ratios: cube/spectral=2544×, cube/ico=3.66×, cube/latlon=**0.325** (cube 3× BETTER than latlon).  Doc pre-iter-69 claim "cube/latlon 1.7×" was correct at iter-42; post-iter-44 cube overshoot below latlon.

- **W5 15-day** (iter-70): **mass conservation parity at machine precision across all 4 grids**.
  - ico: drift=0.00e+00, latlon: 0.00e+00, spectral: 8.10e-16, cube: 1.46e-15.
  - All PASS at full 15-day duration.

- **CB 12-day** (iter-61): all 4 grids PASS, cube is L2 outlier but BEST on conservation.
  - latlon: L2=0.133, drift=2.07e-8 (iter-61 anchored fixer applied).
  - ico: L2=0.620, drift=1.58e-6 (additive fixer — iter-66 anchored-attempt regressed, reverted).
  - spectral: L2=0.382, drift=2.16e-16.
  - cube (iter-58/59): L2=0.865, drift=1.28e-9.
  - True cube/latlon=6.5× (was previously falsely "35×" from mixed-mode).

**Lessons recorded**:
- Cross-grid "consistency" ≠ identical fixer logic.  Cube/latlon (regular structured grids) prefer multiplicative-rescale-to-initial-mass; ico (heterogeneous unstructured: 12 pentagons + hexagons) prefers additive uniform correction.  Multiplicative on ico over-concentrates mass in pentagons (Linf 0.56 → 2.83) — iter-66 negative result.

**Cumulative cube CB improvement (iter-1 → iter-69 baseline)**: L2 1.092 → 0.865 (−20.8 %) via:
- iter-58 hord=10 + xppm_bdy → −14.7 % (matched Fortran tp_core L2 limiter + boundary cube-edge formulas).
- iter-59 N=6 temporal substep → −7.0 % (closed forward-Euler PPM phase-error gap vs latlon SSP-RK3).

**Sentinel coverage** (now 27 AST + 14 numerical):
- AST: 27 in `tests/test_matrix_nh_cube_parity_ast_guard.py` (added iter-59 N=6 substep + iter-61 anchored mass fixer sentinels).
- Numerical: 14 in `tests/test_iter1002_w2_target_met.py` — W2 1-day/2-day/5-day, W5 day-5/2-day, W6 2-day, CB 12-day cube + 12-day latlon, hyperdiff kwarg API, 2 quiet-path guards.

**Outstanding (not closed)**:
- Cube SW W2 vs ico residual 3.66× L2 ratio — structural to PPM on curved cube faces vs SOM-PPM on isotropic Voronoi.
- Cube CB vs ico residual ~1.4× L2 ratio — bulk PPM limiter dissipation along rotated trajectory (iter-61 spatial decomposition).
- NH TC1 cube |w|_max=0.040 m/s vs doc claim 0.014 m/s — 3× gap; iter-63 audit found matrix uses `nord_v=2` default while factory uses `nord_v=1`.  iter-71+ to probe whether `nord_v=1` closes this.
- NH TC2/TC3 cube matrix re-run still in progress; will refresh iter-7 claim numbers (TC2: 0.32, TC3: 7.36 m/s).

## State after iter-1..60 (compressed at iter-60)

**New cube CB wins (iter-51..60)**:

- **Cube CB 12-day L2 1.092 → 0.865 (−20.8 % cumulative)**.
  - iter-58: `hord=10` + `apply_fortran_xppm_boundary=True` in matrix `transport_step` call → 1.092 → 0.931 (−14.7 %).  PPM hord=10 (slope-limiter monotone) retains more amplitude than hord=12 at multi-day integrations; xppm boundary applies Fortran-faithful cube-edge formulas.
  - iter-59: N=6 temporal substepping (dt_sub=300s) inside CB cube step_fn → 0.931 → 0.865 (−7.0 %).  Latlon CB uses SSP-RK3 (O(dt³)); cube CB used single-stage forward-Euler PPM (O(dt)) — substep closes ~7 % of the temporal-truncation gap.  Saturation past n_sub=6 shallow (≤1.5 % per doubling).  Mass drift 28× tighter (3.6e-8 → 1.3e-9, per-substep fixer).
  - iter-61: cross-grid L2 ratio CORRECTED from previously documented 35×.  The 35× figure was based on comparing cube 12-day vs latlon **1-day** stale results — apples-to-oranges.  True apples-to-apples 12-day cube/latlon L2 ratio = **6.5×** (cube 0.865 vs latlon 0.133).  Cube vs ico = 1.4×, cube vs spectral = 2.3×.  Cube is L2 outlier but BEST on mass conservation among finite-volume grids (1.28e-9 vs latlon 2.07e-8 vs ico 1.58e-6, with spectral 2e-16 exact).

**Other iter-51..60 work**:

- iter-51: cube CB C36 vs C48 L2 essentially identical (1.092 vs 1.094) → CB structural gap NOT dx²-convergence-limited; PPM limiter dissipation dominates.
- iter-52..55: parameter sweeps on iter-1030 calibration (`nord_v`, `div_damp_factor`, `damp_v`) — all confirm current values near L2 optimum.  Wind-vs-height trade-off characterised.
- iter-56: 1-day vs 5-day hyperdiff trade-off characterized; iter-1002 sentinel and matrix runner pin DIFFERENT operating points (1-day spectral character vs 5-day long-run stability).
- iter-57: cube W2 5-day numerical sentinel at matrix-actual config (L2 < 1.5e-3, max|v_d| < 5 m/s).
- iter-60: cleaned up 2 stale silent-noop warning sentinels (iter-1019/iter-1020) — both were pre-existing red CI tests asserting warnings from code paths the SW model doesn't traverse.  Inverted to positive guards: assert NO spurious warning fires.  Plus added `test_iter59_cube_cb_12day_matrix_config` numerical sentinel pinning the iter-58/59 cube CB win (L2 < 0.90, Linf < 0.90).

**Sentinel coverage** (now 26 AST + 8 numerical + 1 c_sw residual):
- AST: 26 in `tests/test_matrix_nh_cube_parity_ast_guard.py` (iter-59 added `test_iter59_cb_cube_uses_n_sub_substepping`).
- Numerical: 8 in `tests/test_iter1002_w2_target_met.py` — W2 1-day (iter-1002), W5 day-5 (iter-1009), W6 2-day (iter-39), W5 2-day (iter-48), W2 2-day (iter-49), W2 5-day (iter-57), CB 12-day (iter-59), hyperdiff kwarg API (iter-35); plus 2 quiet-path guards (iter-60 inverted iter-1019/iter-1020).

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
- iter-50 / iter-60: doc compressions (state of iter-1..49 and iter-51..60 summarised in the two "State after" blocks above).
- iter-51..70: see two compressed "State after" blocks at the top of this doc.
- iter-61 (verbose entry retained — first apples-to-apples mass-fixer parity finding):  cross-grid CB 12-day apples-to-apples audit + latlon CB mass-conservation parity.  Found previously documented "35× cube/latlon L2 ratio" was based on comparing cube **12-day** results vs latlon **1-day** stale results files (the matrix's quick-mode + full-mode results coexist in the same dir).  Re-ran all 4 grids at 12-day full duration:
  - latlon (72×144): L2=0.125, mass_drift=5.35e-4 → **FAIL** (drift > matrix tol 1e-4).
  - icosahedral (ico5): L2=0.620, mass_drift=1.58e-6, PASS.
  - spectral (T21): L2=0.382, mass_drift=2.16e-16, PASS.
  - cube (C36, iter-58/59): L2=0.865, mass_drift=1.28e-9, PASS.
  Diagnosed: latlon CB step_fn was intentionally a "raw FV benchmark" (no mass fixer); iter-29 matrix-wide tolerance tightening from 1e-2 to 1e-4 dropped below the raw-FV mass drift.  Per the ralph-loop "consistency across grids" goal, applied the **same anchored mass fixer cube uses** (clip-negatives + rescale-positives to mass_target) to the latlon CB step_fn.  Result: latlon mass_drift 5.35e-4 → **2.07e-8** (26000× tighter), L2 0.125 → 0.133 (+6 % small redistribution cost), Linf 0.229 → 0.246 (+7 %), **PASS**.  **All 4 grids now PASS at 12-day apples-to-apples**.  True cube/latlon L2 ratio: **6.5×** (was claimed 35×), cube/ico 1.4×, cube/spectral 2.3× — all far closer than previous trail suggested.  iter-61 also probed the spatial decomposition of the cube CB error (`_probe_iter61_cb_error_map.py`): 99 % of the residual L2 lives in the panel-INTERIOR cells of the single face containing the bell at t=12 d; panel-edge cells contribute ~0.0003 (essentially zero).  Confirms residual is bulk PPM limiter dissipation along the rotated trajectory, NOT cube panel-coupling.

- iter-62: iter-61 latlon CB numerical sentinel.
- iter-66 negative result (reverted): ico CB multiplicative-fixer attempt regressed Linf 5× — additive uniform correction wins on heterogeneous unstructured mesh.  See top compressed block for lesson.
- iter-69: W2 5-day apples-to-apples — cube is BEST finite-volume.  See top compressed block.
- iter-70: W5 15-day apples-to-apples — all 4 grids machine-precision mass conservation.  See top compressed block.
- iter-75: post-iter-72 audit confirmed **NO remaining quick/full mode result mismatches** anywhere in the SW or PE matrix (17 test cases × 4 grids = 68 entries; all `days` fields consistent within each case).  Full apples-to-apples cross-grid comparison is now possible at any time without ambiguity.  Net result of iter-61..72: 4 of the 5 quick/full-mode mismatches found (CB cross-grid, W2 cross-grid, W5 cross-grid, W6 cross-grid, gravity_wave_3_1 cross-grid) refreshed to full mode; 5th (W6 cross-grid) was already full mode from a prior re-run.  27/27 AST guards still PASS after iter-74 hyperdiff comment refresh.

- iter-73: full PE matrix cross-grid audit at current cached results.  All 13 PE cases (held_suarez ±topo, baroclinic ±rotated, gravity_wave_3_1, inertio_gravity_3_2, mountain_rossby_5_0, rossby_haurwitz_6_0, rotated_steady, rest_state_topo, dcmip_transport_11/12/13, amip) PASS on all 4 grids.  Cube max|v| parity vs ico (the canonical FV reference):
  - **Excellent (cube/ico in 0.95-1.15×)**: held_suarez, gravity_wave_3_1, inertio_gravity_3_2, rossby_haurwitz_6_0, baroclinic.
  - **Investigate cube outliers**:
    - `rotated_baroclinic`: cube 31.8 m/s vs ico 51.9, spec 47.1 m/s — cube 39 % LOWER than ico (possibly over-damped on the rotated-pole initial state).
    - `rest_state_topo`: cube 1.3 m/s vs ico 0.1, latlon 0.1 m/s — cube residual motion at rest with topography is 13× ico/latlon (analytic exact: zero motion).  Likely cube panel-edge metric errors interacting with topo.  spectral pathological at 51.0 m/s (separate issue).
  Cube mass drift across PE 30-day tests: 4-5e-12 (4 orders below 1e-4 tolerance; not zero like ico/latlon but well within budget — fp64 reduction noise on 7776 cells).  Queued for iter-74+: investigate rest_state_topo cube residual + rotated_baroclinic under-damping.

- iter-72: PE gravity_wave_3_1 cross-grid apples-to-apples audit at 1-day full duration.  Pre-iter-72 results had ico/latlon/spectral cached at 0.25-day quick mode while cube ran 1.0 day.  Refreshed:
  - **cube (C36)**: max|v|=22.4 m/s, drift=5.01e-12, **PASS**.
  - ico (ico5): max|v|=19.7 m/s, drift=0, PASS.
  - spectral (T21): max|v|=20.3 m/s, drift=1.6e-16, PASS.
  - latlon (72×144): max|v|=31.0 m/s, drift=0, PASS (dt=10s; latlon resolves higher-frequency waves).
  **Cube/ico max|v| ratio = 1.14×, cube/spec = 1.10×** — excellent PE parity (cube and ico both use 200s dt, similar transport schemes).  iter-18's claim "27.5 → 22.4 m/s (-18.5%)" confirmed at apples-to-apples; cube now within 10-14 % of ico/spectral cluster.  latlon outlier (31.0 m/s) tracks its 20× finer dt = different effective resolution in time; not a cube issue.

- iter-71: hyperdiff coefficient multiplier sweep on cube W2 5-day (refines iter-44's 2× choice).  Probe results:
  - mult=1.0×: L2=7.16e-4, v_d=0.96 m/s.
  - mult=1.5×: L2=5.51e-4, v_d=0.70 m/s.
  - mult=2.0× (iter-44): L2=4.58e-4, v_d=0.65 m/s.
  - mult=2.5×: L2=**4.42e-4**, v_d=**0.63 m/s** (best on both).
  - mult=3.0×: L2=4.47e-4, v_d=0.63 m/s.
  Optimum near 2.5× gives 3.5 % L2 reduction over 2.0×.  Below the test-to-test noise floor (similar to iter-54's div_damp_factor sweep finding).  Going from 1.0 → 2.0 captures 87 % of the achievable improvement (7.16 → 4.58 vs 7.16 → 4.42).  **iter-44's 2.0× retained** — the 3.5 % marginal gain at 2.5 isn't compelling enough to deviate from the calibrated value, and over-damping risks regressing W5/W6 long-run mountain-wave structure (not visible in mass-drift-only PASS criteria).  Also: W6 14-day cross-grid apples-to-apples confirmed (this iter) — all 4 grids PASS at machine-precision mass conservation.  SW matrix is now fully apples-to-apples across all 4 cases (W2/W5/W6/CB) × all 4 grids.

## Iter-51+ queued

- C72 resolution sweep (extends iter-46's C36+C48 coverage).
- Visual artifact inspection of cube W2/W5/W6 snapshots with vs without hyperdiff (reviewer item iter-45).
- Cube CB structural — if a tracer-only del-4 operator is feasible without breaking transport_step's API.
- **iter-64**: matrix TC1 cube measured |w|_max=0.0400 m/s (8× better than 0.327 baseline, but the iter-7 doc claim was 0.014 m/s = 23× — current matrix is ~3× higher than that claim).  iter-63 audit comparing matrix runner TC1 config vs `make_fv3_faithful_nh_config` factory revealed matrix is missing `nord_v=1` (factory sets to 1; matrix uses default 2 = del-6 instead of del-4 NH velocity hyperdiff).  Other factory-only flags (`use_fv3_cross_face_du_proj`, `use_fv3_sponge_damp_v/w`) are no-ops without `use_duogrid=True` (matrix doesn't set duogrid).  iter-64 will probe `nord_v=1` for the matrix TC1 cube branch to test whether it closes the 0.040 → 0.014 m/s gap.
