# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/` (sw_core.F90, tp_core.F90, dyn_core.F90, fv_dynamics.F90, nh_core.F90, nh_utils.F90, fv_mapz.F90, a2b_edge.F90).

Scope: get FV3 cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State snapshot (iter-1)
- Main has 64 iters of mass-conservation hardening (anchored fixer + fp64 budget across `core/conservation.py` + `diagnostics/`) merged via PR #259.
- Codex FV3 fidelity branch `codex/fv3-fortran-fidelity-ralph-loop-20260417` has 1697 commits of cube-vertex audit + calibration work; NOT all merged. Notable findings still relevant:
  - iter-968/969/975/976/977: 14 routines (c_sw, d_sw1, d_sw5, d_sw6, divergence_corner_duo, d2a2c_vect, xppm/yppm, del6_vt_flux, compute_transport_quantities) audited bit-faithful to FV3 Fortran.
  - iter-971/972: ported FV3 `a2b_ord4` 4th-order cell→corner; wired into d_sw5 Smagorinsky branch (`_interp_center_to_corner_a2b_ord4` in `core/fv3_sw_core.py`).
  - iter-981/982/983: residual cube-vertex bug isolated to c_sw + p_grad_c imbalance on polar-face cube vertices (face=4 i=35 j=35 N-pole NE, face=5 i=35 j=1 S-pole SE).
  - iter-1000-1009-1021-1030: SW cube W2/W5 calibration evolution
    - iter-1009: `(div=10·cube, damp_v=0.04)` → W2 v_ll=0.1147 m/s, W5 day-5 spd=68.6 m/s
    - iter-1021: `(9, 0.035)` → v_ll=0.1137, W5 spd=53.3
    - iter-1030: `(8, 0.030)` → v_ll=0.1138 m/s, W5 spd=45.1 m/s (BEST W5, sentinel-pinned)
- Matrix runner SW W2/W5 cube config (`scripts/run_atmosphere_test_matrix.py:2006-2012`) is at iter-893 baseline `damp_v=0.06`; iter-1030 sentinel-pinned best is `damp_v=0.030`.

## Iter-1 action
Bring matrix runner W2/W5 SW cube config to iter-1030 calibration (`damp_v=0.030`).
Rationale: matches the calibration the iter-1002/1030 sentinel pins as the W5-best balanced calibration. Reduces vorticity damping → frees the 2nd-order divergence damping to suppress cube-vertex mode at lat ±35° without over-damping kinetic energy in long W5 runs.

Verification gate: run the W2 cube matrix test at C36 5 days. Must PASS.

## Open cube-parity items (rolling)
1. **Cube-vertex c_sw + p_grad_c imbalance** (codex iter-983): 1.3 m/s spurious uc, 0.5 m/s spurious vc on W2 IC. Localised to polar-face vertices. Source under investigation in codex branch; not yet root-caused.
2. **W5 long-run stability** (codex iter-1003-1004): cube-edge polar mechanism degrades W5 past day 5+ even at best calibration.
3. **Cube SW W2 v_ll_Linf** ≈ 0.114 m/s at iter-1030. Latlon W2 v_ll typically ~10× tighter; spectral exact to truncation. Parity gap = ~1 order of magnitude.
4. **Cosine-bell matrix gate disparity**: latlon CB uses `_DYCORE_MASS_DRIFT_TOL_CB=1e-4` (raw transport), cube CB uses anchored fixer + standard `1e-6`. Apples-to-apples comparison requires running both with/without the anchor.
5. **PE/NH cube parity**: not yet probed.

## Iter trail
- iter-1 (2026-05-13): scope set, state snapshot, matrix runner SW cube W2/W5 config bumped to iter-1030 calibration (`damp_v=0.030`). `test_iter1002_w2_v_ll_linf_meets_target` PASS, `test_iter1009_w5_day5_artifact_free` PASS. Open: verify W2 5-day + W5 15-day matrix paths (the calibrations were sentinel-tested at C36 1-day / day-5 only).
- iter-2 (2026-05-13): probed cube CB raw transport (anchor stripped from `transport_step` in the matrix CB cube path) to assess apples-to-apples vs latlon raw-FV benchmark.  Result: cube CB at C36, 48 × dt=1800s (1 day) drifts `mass_drift = 3.82e-4` vs latlon raw `1.49e-5` (~25x worse).  Per-step breakdown shows the drift is NOT linear in step count — step 1 = 6e-10, step 30 = 8e-10, step 40 = 2.3e-7, step 48 = 1.9e-4 → exponential blow-up between step 40 and 48.  Root cause: PPM monotone limiter near cube vertices creates small negative h cells which subsequent transport amplifies; anchor's positivity clip (`jnp.maximum(h_new, 0.0)`) suppresses these.  Apples-to-apples cube↔latlon requires fixing the per-step flux mismatch at cube panel edges (not just removing the anchor).  Revert applied; matrix-runner CB cube anchor restored.  iter-1 W2 5-day matrix verification: **PASS** at iter-1030 calibration (C36 5-day: L2=2.45e-3, Linf=2.40e-2, v_ll_Linf=3.65 m/s post-regrid, 25.5s wall).  iter-1030 → matrix is safe.  Open for iter-3+: trace flux closure gap at panel edges in `compute_transport_quantities` (sin_sg halo) + cube-vertex c_sw + p_grad_c imbalance (codex iter-983: face=4 i=35 j=35 polar-face NE vertex).
