# FV3-faithful cubed-sphere — change log (shrunk @ iter94; prior detail in git)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
matching MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge artifacts**, visual +
quantitative, codex/oracle-reviewed. CPU only (Metal broken). **User directive: never A-grid;
be FV3-faithful; use the Fortran as oracle; do NOT improvise.** Branch `latlon-fv-amip-verify`.

## ✅ DONE / VALIDATED / codex-reviewed
- ATM panel-edge imprint: cc→D-grid scalar lift (baroclinic v_rms 4.95→0.60). Cross-grid align
  (`_canvas_lat/lon`). OCEAN geostrophic 40%→1.45% (cube barotropic → FV3 SW `fv3sw`, never A-grid).
- gnomonic_ed grid (FV3 operational gt=0) WIRED + halo-collapse fixed (iter73, separable 1D angle)
  + codex-approved. `create_cubed_sphere(gnomonic="ed")`; equiangular byte-identical. ed bounds the
  C96 W5 eigenmode (saturates ~115, no NaN) but doesn't eliminate it.
- **area_corner FV3 (#faces)-junction SCALING DONE (iter84-91, codex-APPROVED, NET-ZERO regression):**
  edges ×2 (≈0.865·interior), vertices ×3 (≈0.675·interior), C1 guard n>=1. Replaces the iter-670
  interior-copy (which mirrored FV3's *halo* extrapolation, not in-domain grid_area). Oracle:
  fv_grid_tools.F90 edge=2*get_area (976-1033), vertex=3*get_area (1036-1067). Regression test
  C1/C12/C36 all edges+vertices. W2 L2=1.76e-4 unchanged, ocean 9/9 rest machine-zero.
  - **CHORD caveat (iter91):** legoESM `sg_area` is PLANAR chord (not FV3 spherical get_area). Only
    the ×2/×3 SCALING is FV3-faithful; absolute area is an O(dx²) chord approx. Spherical get_area
    gives C1=0.75 (vs chord 0.659) + trips the whole chord-era SW gold-file surface → TRACKED
    follow-up (needs gold regen), NOT done. All test/comment wording corrected to say so (codex).
  - **da_min_c insight:** `da_min_c=min(1/rarea_c)` (FV3 global_mx_c, fv_grid_utils.F90:743). The
    faithful smaller vertex area lowers da_min_c → FV3-correct (weaker) divergence damping. Shifted
    4 SW-core gold fingerprints (nord0/nord1/smag via da_min_c; corner-vort boundary-local) — all
    re-pinned + verified (commit 21b12082). DISPROVES "area fix damps the eigenmode" — it exposes it.

## EDGE-ARTIFACT STATUS
Reliable metric = `compute_cross_face_continuity` (halo.py). BAROCLINIC clean (C36→C72 ≤1.24×).
SW physical fields continuous. **OPEN — C96 W5 edge eigenmode (production RK3):** max|wind| ~38 to
day3.5 then blows 38→79 (faces 0,4); CFL-independent, mass-conserving. Working harness (iter86b):
W5 C96 div_damp=8/hyperdiff=2·hdc reproduces recorded 26/32/33/40/80. Root = SW-core vertex-vorticity
(NOT grid, NOT area — both proven NEUTRAL/exposing). v-imprint W2 v_ll_Linf=0.344 (~0.9%, faces).

## FAITHFULNESS SCORECARD
FAITHFUL: gnomonic_ed; get_area (the function); upwind vort-flux `_vorticity_flux`/`fv_tp_2d`
(but only in the FB path). MINOR: a2b_ord4 (inert); cross-face vector halo ORTHOGONAL (drops FV3
1/sin_sg(5), O(cosθ) seam err). **MAJOR (the convergent root, iter92):** BOTH production SW
(operators_cdgrid.py:1167-1185) AND 3D PE (primitive_eq_cdgrid.py:445) use CENTERED `zeta_corner*v_d`
vorticity advection (energy-conserving but dispersive), NOT FV3's UPWIND donor-cell flux
(`_vorticity_flux`, sw_core.F90:416-480: fy1=(v_d-uc*cosa_u)/sina_u contravariant, vort_x=upwind).
Production SW = co-located Arakawa-Lamb (not staggered c_sw→d_sw), so the upwind flux can only enter
via the FB chain. ONE gap plausibly explains all 3 symptoms: C96 W5 eigenmode + cosine_bell 5× +
v-imprint (no implicit upwind dissipation at vertices).

## FB CHAIN (FV3 c_sw→d_sw FB; the faithful staggered scheme; EXPERIMENTAL, residual open)
`fv3_fb_sw_step`/`FV3FBShallowWaterModel`. Production = RK3 Arakawa-Lamb (works, "algorithmically
faithful"; FB = deferred upgrade). Fixes done: BUG1 missing D-grid backward PGF → Phase4 one_grad_p
(commit 1284e71b); BUG2 needs DUOGRID (ng=3) → W2 C36 4h→33.6h (8.5×). **RESIDUAL open:** W2 C36
day1 max|u_d|=48.6 (vs 38.6), day2 NaN. RULED OUT as residual root: dissipation (iter82), Phase4 a2b
(iter81), wind-halo/damping (iter83), **corner-vorticity METRIC halo (iter94: edge-copy vs O(dx²)
linear-extrap dxc/dyc → <0.2% Δ, 48.60→48.53)**. REMAINING candidates: `_corner_vorticity` uc/vc halo
RECONSTRUCTION (2-pt center-avg+re-stagger, fv3_sw_core.py:1189-1208); d_sw zeta (covariant
circulation, no corner correction, _d_sw_native:1885); vorticity-flux/d2a2c corner. Residual is a
STRUCTURAL vertex-vorticity growth mode (dissipation/metric-independent).

## VERIFICATION (current HEAD)
- Cross-grid SW 16/16 all grids (iter92): W2 L2 cube 1.76e-4 | latlon 2.67e-4 | ico 9.9e-5 | spec
  3.6e-8 → cube FAITHFUL ~latlon/ico. cosine_bell L2 cube 0.131 vs latlon 0.025 (~5×) → advection gap.
- test_cdgrid_fv3_regression.py: HEAD 17 failed == pre-area-change baseline 9c78db6c (17). area work =
  NET-ZERO. The 17 (cosine_bell, production_tendencies, d_sw_native, mutation suites, source/file
  sentinels) are PRE-EXISTING (deleted results/*.png + concurrent-session churn) — separate cleanup.
- Visual PNGs surfaced to user (W2 v-wind, W5 wind_speed, cross-grid comparisons) — human verdict pending.

## ACTION QUEUE (FV3-faithful, oracle-driven, no improvise)
1. SW-core vertex-vorticity eigenmode = the real edge-artifact root. Path = the FB chain (faithful
   upwind staggered). NEXT: FB residual — test the uc/vc reconstruction; or isolate via FB-vs-prod
   single-step tendency diff on smooth W2. Then FB→production (gated, eigenmode-validated).
2. cosine_bell advection 5× latlon (production accuracy gap) — investigate cube tracer/vorticity transport.
3. chord→spherical sg_area upgrade (tracked, needs gold regen). 4. 3D PE upwind PPM vort. 5. thread
   FV3 sin_sg(5) seam rotation. 6. pre-existing 17-test swamp. 7. human visual verdict. 8. ocean/AMIP
   cross-grid (climate CPU-infeasible). State: memory `cube-fv3-faithfulness-state`.
