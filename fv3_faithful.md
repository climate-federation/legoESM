# FV3-faithful cubed-sphere — change log (shrunk @ iter ~60)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`,
70 F90), matching MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge
artifacts**, visual+quantitative, codex/oracle-reviewed. CPU only (Metal broken).
**User directive: never A-grid; be FV3-faithful; use the Fortran as oracle; do NOT
improvise.** Branch `latlon-fv-amip-verify`, ~70 commits ahead. Per-iter detail in git.

## ✅ FIXES (done, validated, codex-reviewed)
- ATM panel-edge imprint: cc→D-grid lift (u,v) as scalars, 7 sites
  `center_to_dgrid_vector`/`pad_halo_vector_4d`. baroclinic v_rms 4.95→0.60, mass→1e-15.
  Test `test_vector_cc_to_dgrid_wind_lift.py`.
- CROSS-GRID alignment: one canonical canvas `_canvas_lat/_canvas_lon` for regrid+npz+
  axes (fixed tens-of-deg translation + node/cell drift; also unblocked latlon SW).
  Test `test_latlon_regrid_alignment.py`.
- OCEAN geostrophic non-zonal artifact 40%→1.45%: cube barotropic was A-grid → routed
  through FV3 SW core `barotropic_substeps_fv3sw` (`barotropic_staggering="fv3sw"`,
  never A-grid). RE-VERIFIED iter55 fresh: T drift 7.5e-14, max_speed 0.0143≈latlon/mpas.
  Test `test_fv3sw_barotropic.py`. Ocean FC viscosity vector-halo (4fe7108c).
- Ocean barotropic faithful option: `barotropic_staggering="fv3edge"` (TRUE FV3 edge-
  staggered SW core), gated, validated 8/9 (phillips needs non-latlon div_damp → fv3sw
  stays 9/9 default). Test `test_fv3edge_barotropic.py`. Codex-approved.
- [iter57] Fixed stale iter-707 KE-fingerprint test: `places=4` abs (~5e-10 rel on 1e5)
  spuriously failed on 6e-9 float drift → rtol=1e-6, L2-scaled for the cancelling sum.

## EDGE-ARTIFACT TRUTH (iter57-59; corrected from earlier "PROVEN clean" overstatement)
Reliable metric = TRUE cross-face continuity via the model's real `pad_halo_4d`
(cross-seam first-diff RMS / same-face interior first-diff RMS; real discontinuity=5-50×).
Committed as `compute_cross_face_continuity` (halo.py) + `tests/grids/
test_cross_face_continuity_metric.py` (3/3). SUPERSEDES the pooled same-face
`compute_edge_artifact_metric` (halo.py:2994), which amplifies edge curvature and is
UNRELIABLE both ways (gave 5.95× on W5 v where true continuity is 1.30×).
VALIDATED against a REAL artifact (iter61): on C96 W5 the metric tracks the eigenmode
onset in lockstep — wind_speed cross-seam holds 1.57-2.22 while stable (day0-3.5), then
rises 2.29→2.76→3.07 as the eigenmode blows up (day4-5). Cleanly separates clean (≤2.2)
from artifact (>2.5). So the metric detects genuine edge artifacts, not only synthetic ones.
CODEX-REVIEWED iter61: adversarial-review caught a real denominator bug (interior gradient
`gi` included the N/E halo row `a[n+1]-a[n]` = a seam jump → self-normalizing, suppressed
N/E-edge detection). Fixed to interior-only diffs + added a constant-per-face guard test
(interior grad=0 ⇒ ratio must be huge; the bug collapsed it to ~1). Re-review: APPROVED.
Clean fields unchanged (W2 u 1.56-1.60); eigenmode detection sharper (3.07→3.38).
- BAROCLINIC (3D PE): genuinely clean, converges C36→C72. v-roughness ≤1.24× incl
  rotated_baroclinic (jet crosses seams).
- SHALLOW-WATER: all PHYSICAL fields CONTINUOUS — C36/C48/C96 cross-face FLAT: W2 u
  1.57→1.56→1.55, height 1.62→1.60→1.61; W5 v 1.30, wind_speed 1.58, height 2.19 (C36).
  No large edge artifact. (Error-field W2 v 2.23× is the sub-0.34 m/s residual.)
- **OPEN BUG — C96 W5 edge eigenmode:** W5 (mountain) max|wind| tracks C36/C48 (~38 m/s)
  to day3.5 then BLOWS UP 38→51→78.9→62, edge/corner-localized (faces 0,4). C36/C48 stay
  flat. Matches the runner's documented "C96 eigenmode" that dt-tuning has chased.
  [iter61 — CFL REFUTED] C96 W5 at dt=300/150/100 all blow up to ~80/77/76 m/s at day5
  (dt-INDEPENDENT) ⇒ NOT a CFL/dt-scaling issue; a genuine grid-scale SPATIAL eigenmode.
  ORACLE root-cause + verify-first (iter59): FV3 controls grid-scale edge modes via
  adaptive-Smagorinsky `dddmp` (oracle `fv_arrays.F90:360` → operational 0.2; hi-res
  `nord=1,d4_bg=0.075`). legoESM sets `dddmp=0.0` + pragmatic aggregate `div_damp` +
  biharmonic `hyperdiff` (tuned C16-C48). TESTED `dddmp=0.2`: byte-identical (NO effect)
  → the FV3 d_sw5 fields are INERT in the production `FV3EdgeShallowWaterModel` (`.step`
  uses `fv3_sw_tendencies` RK3, not `_d_sw_native`); they live only in the experimental/
  unstable FB chain (`fv3_fb_sw_step`, ~50 steps→NaN). Live knobs: `div_damp=16×`→NaN,
  `hyperdiff=6×`→stabilizes C96 BUT is improvising (scalar biharmonic, not FV3) → NOT
  committed. FAITHFUL fix = stabilize+wire the FB chain so FV3's dddmp reaches production
  = FOUNDATIONAL. (Hypothesis falsified by experiment, not asserted.)
- [iter60 — FB-chain diagnosis, oracle-aligned] Reproduced + classified the FB chain
  (`fv3_fb_sw_step`) on W2 C24, dt=150, jitted: contrary to the "~50 steps→NaN" label it
  is STABLE 150 steps (no NaN), BUT `max|h|` drifts UP monotonically 3104→4630 (~0.3%/step)
  where W2 should hold ~2998. Classified: **total_mass drift = 0.00 (machine-zero, EXACTLY
  conserved)** and max|h| is ALWAYS at the **i=0 cube EDGE** (row 0 of a face). ⇒ the FB
  blocker is an EDGE-LOCALIZED, MASS-CONSERVING amplitude instability — height piles up at
  seams while mass stays exact. SAME cube-edge eigenmode as the C96 W5 production blow-up.
  `dddmp=0.2` barely changes it (4630→4506) ⇒ NOT a divergence-damping problem; and mass
  conserves ⇒ NOT a flux-conservation bug. Root cause = the EDGE/SEAM treatment → the
  orthogonal cross-face vector rotation that DROPS FV3 `sin_sg(:,:,5)` (scorecard). FAITHFUL
  FIX = FV3 non-orthogonal seam rotation `pad_halo_dgrid_vector_4d` /
  `use_fv3_cross_face_du_proj` (12/12 tests, currently gated OFF; that flag lives on the
  NH `compressible_euler_cdgrid`, NOT the SW FB chain → wiring is a real change).
- [iter61] Exposed-flag ablation on the FB h-drift: `apply_fortran_xppm_boundary=True`
  (FV3 edge PPM), `apply_legacy_d_sw5_corner_corrections=True`, and (earlier) `dddmp=0.2`
  ALL give BYTE-IDENTICAL h-drift (3228/3801/4630 @50/100/150). ⇒ the edge mode is in the
  CORE seam mass-transport / wind-convergence, NOT any exposed FV3-faithfulness knob.
  (Wind-damping `hyperdiff` controls it indirectly = the improvise path, rejected.)
- [iter61 — seam-rotation hypothesis WEAKENED, verify-first] `_d_sw1_recompute_ut_vt`
  (the FB chain's d_sw1) ALREADY uses the non-orthogonal `cosa_u/cosa_v/rsin_u/rsin_v/
  sin_sg` metrics — so the FB h-drift is NOT a naive-orthogonal-rotation bug. (The
  orthogonal `pad_halo_vector_4d` sin_sg(5)-drop is in the PRODUCTION RK3 primitive-eq
  path, a separate issue.) Every exposed knob (dddmp/xppm/d_sw5_corner) AND structural
  hypothesis (seam rotation) on the FB edge mode has been FALSIFIED or shown inert. The
  drift is a subtle CORE edge mode — candidates: c_sw 1st-order upwind mass diffusion at
  the seam, or the iter-947 duogrid OLD/NEW halo delta. Pinpointing it = methodical
  d_sw-substep-vs-FV3 line-by-line comparison = FOUNDATIONAL (multi-day), NOT a loop
  probe. FB JIT probing STOPPED (~min/compile, diminishing returns; per user no-improvise,
  the faithful fix is the core c_sw/d_sw rebuild, not a damping/rotation patch).
- STILL MISSING: human visual PNG inspection (assistant barred from Read images):
  `results/atmosphere/shallow_water/williamson{2,5}/cubed_sphere/C36/snapshots_{v,wind_speed}_native.png`.

## FAITHFULNESS SCORECARD (12-agent oracle workflow iter57; Fortran-vs-port direct read)
**FAITHFUL but ORPHANED (not on the running model):**
- gnomonic_ed grid: bit-faithful (1 ULP vs independent numpy FV3 reimpl) but ZERO prod
  callers — `create_cubed_sphere` runs EQUIANGULAR (grid_type=2).
- get_area/cell_center3: faithful; prod `grid.area` uses equivalent l'Huilier (rel<1e-6).
- upwind vort-flux/`fv_tp_2d`/xppm-yppm: line-for-line faithful but orphaned (3D PE
  doesn't call it). Latent `hord==9` iv-mislabel (fv_tp_2d.py:504; unexercised).
**MINOR (partially wired):**
- `d2a2c_vect`: LIVE duogrid 4th-order path faithful; non-duogrid corner overrides off.
- `a2b_ord4` zeta_corner: bit-faithful interior; inert by default.
- cross-face vector halo `pad_halo_vector_4d`: ORTHOGONAL — drops FV3 `1/sin_sg(:,:,5)`
  non-orthogonality → real O(cosθ) seam error. Faithful `pad_halo_dgrid_vector_4d`
  (12/12 tests) gated behind `use_fv3_cross_face_du_proj=False`.
**MAJOR gap:**
- SW core: production `CDGrid`/`FV3Edge` is single-stage SSP-RK3 over ONE Arakawa-Lamb
  centered tendency, NOT FV3 forward-backward c_sw→d_sw. Faithful `fv3_fb_sw_step` exists
  but EXPERIMENTAL/unstable (~50→NaN), unreachable from factory. ⇒ "SW faithful" applied
  only to the FV3Edge operators, not the production time-split.
- 3D PE D-grid vorticity/KE: LIVE non-orphaned gap. Centered `zeta_corner*v_d` (primitive_
  eq_cdgrid:445), NOT FV3 upwind PPM `hord_vt`. Docstring honestly "Research path".

## GRID gap
legoESM cube = EQUIANGULAR gnomonic (grid_type=2). FV3 operational = `gnomonic_ed`
(grid_type=0, √2 dx-ratio). Port + `make_fv3_native_grid` done+tested but orphaned;
wiring needs full metric/halo re-derivation (`compute_padded_angle/half_metrics`,
`_compute_exact_cell_areas`, C-D supergrid all hardcoded equiangular) + re-calibration.
Foundational.

## VERIFICATION
- SW 16/16 all grids ≤1 cell aligned. Full fast cube atm dynamical suite PASS + NH
  dcmip_tc1, mass machine-zero. Ocean matrix cube 9/9; rest_state ×12 machine-zero.
- Cross-grid CLOSENESS (W2 height L2 vs exact): cube 1.76e-4 BETWEEN ico 9.9e-5 and
  latlon 2.67e-4 (cube more accurate than latlon). gravity_wave 5%, geostrophic/phillips/
  barotropic match. Regression-clean 44/44. Health 23/24 (1 fail = collaborator's latlon
  `implicit_cn`, not cube scope).

## ACTION QUEUE
SAFE-BOUNDED (no risk to equiangular default):
- [TODO] Numerical regressions locking orphaned faithful ports (gnomonic_ed vs numpy FV3
  ref; a2b_ord4 vs hand FV3 cascade). Fix latent `hord==9` mislabel.
- [TODO] Human visual PNG inspection (paths above).
FOUNDATIONAL (multi-day, re-calibration; NOT a bounded loop iter — per user, no improvise):
- **Stabilize + wire the FB chain** (`fv3_fb_sw_step`): brings FV3's c_sw/d_sw 2-stage +
  dddmp Smagorinsky into production SW → fixes BOTH the SW-2-stage gap AND the C96 W5
  eigenmode. Blocker = FV3 del2/del4 dissipation at the c_sw/d_sw phases. [highest value]
- 3D PE upwind PPM vorticity transport (replace centered zeta_corner*v_d).
- Wire gnomonic_ed grid (full metric/halo re-derivation + re-calibration).
- Thread FV3 non-orthogonal seam rotation (sin_sg(5)) into `pad_halo_vector_4d`.
- Climate-equilibrium cross-grid (HS/AMIP): CPU-INFEASIBLE here (HS needs ~200-day spin-up).
State persisted: memory `cube-fv3-faithfulness-state`.
