# FV3-faithful cubed-sphere — change log (shrunk @ iter ~79)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`,
abs `/Users/pierregentine/Documents/Code/FV3/atmos_cubed_sphere-symmetryclean/model/`),
matching MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge artifacts**,
visual+quantitative, codex/oracle-reviewed. CPU only (Metal broken).
**User directive: never A-grid; be FV3-faithful; use the Fortran as oracle; do NOT
improvise.** Branch `latlon-fv-amip-verify`. Per-iter detail in git.

## ✅ FIXES (done, validated, codex-reviewed)
- ATM panel-edge imprint: cc→D-grid lift (u,v) as scalars. baroclinic v_rms 4.95→0.60,
  mass→1e-15. Test `test_vector_cc_to_dgrid_wind_lift.py`.
- CROSS-GRID alignment: canonical `_canvas_lat/_canvas_lon`. `test_latlon_regrid_alignment.py`.
- OCEAN geostrophic 40%→1.45%: cube barotropic A-grid → FV3 SW core `fv3sw` (never A-grid)
  + FC viscosity vector-halo. Faithful upwind option `fv3edge` (gated, 8/9).
- [iter62] `hord==9` iv mislabel → `_pert_ppm_iv0` (FV3 iord=9=iv=0).

## EDGE-ARTIFACT TRUTH (iter57-61; earlier "PROVEN clean" was OVERSTATED)
Reliable metric = `compute_cross_face_continuity` (halo.py): cross-seam/interior first-diff via
real `pad_halo_4d` (≈1 continuous, 5-50× = real jump). SUPERSEDES pooled `compute_edge_artifact_
metric`. Codex caught+fixed a denominator bug; re-APPROVED. Tests 4/4.
- BAROCLINIC (3D PE): genuinely clean, converges C36→C72 (≤1.24× incl rotated).
- SHALLOW-WATER: all PHYSICAL fields CONTINUOUS — C36/C48/C96 FLAT. No large edge artifact.
- **OPEN — C96 W5 edge eigenmode (production RK3):** max|wind| tracks C36/C48 (~38) to day3.5 then
  blows 38→79 (edge/corner faces 0,4). CFL-independent; mass-conserving. gnomonic_ed partially
  fixes (below). Root suspected = c_sw/d_sw faithfulness (see FB chain).
- STILL MISSING: human visual PNG inspection (assistant barred from Read images):
  `results/atmosphere/shallow_water/williamson{2,5}/cubed_sphere/C36/snapshots_{v,wind_speed}_native.png`.

## FAITHFULNESS SCORECARD (12-agent oracle workflow iter57)
**FAITHFUL:** gnomonic_ed grid (now WIRED, below); get_area/cell_center3; upwind vort-flux/`fv_tp_2d`.
**MINOR:** `a2b_ord4` (inert default); cross-face vector halo `pad_halo_vector_4d` ORTHOGONAL —
drops FV3 `1/sin_sg(5)` (O(cosθ) seam err); faithful `pad_halo_dgrid_vector_4d` (12/12) gated off.
**MAJOR:** SW core prod = single-stage SSP-RK3 (1 Arakawa-Lamb tendency) NOT FV3 FB c_sw→d_sw
(FB chain progressing, below). 3D PE D-grid vort/KE = centered `zeta_corner*v_d`
(primitive_eq_cdgrid:445), NOT FV3 upwind PPM `hord_vt`.

## FB CHAIN (FV3 c_sw→d_sw forward-backward; the convergent root for eigenmode + algo-faithfulness)
Production uses RK3 Arakawa-Lamb (works). `fv3_fb_sw_step` / `FV3FBShallowWaterModel`
(EXPERIMENTAL) ports the true FV3 2-stage FB. iter76-78 progress (W2 C36, the simplest steady case):
- **Instability is a dt-INDEPENDENT GROWING MODE, not tuning** (NaN at ~3h regardless of dt=300/200/
  100; zero-dissip ≡ FV3-dissip; heavy dissip worse). ⇒ a coupling BUG vs the oracle, NOT dissipation.
- **BUG 1 (FIXED, committed 1284e71b): missing D-grid backward pressure gradient.** The FB step
  applied the PGF only at the C-grid (`_p_grad_c` on uc/vc); the prognostic D-grid winds u_d/v_d
  never felt -∇Φ → no geostrophic restoring. Added **Phase 4 = FV3 `one_grad_p`** (dyn_core.F90:2347):
  `gz_b=a2b_ord4(g*(h_new+h_s))` corners, `u_d += dt*rdx_u*(gz_b[:,:-1]-gz_b[:,1:])` (+v), backward
  on h_new, dt-linear (ke_corner ∝ dt verified). t=0 decomposition: PGF opposes the d_sw residual
  (corr -0.924, ratio 0.93), cuts the one-step imbalance 2.6× (rms 0.330→0.126); flipped sign worse.
  11-agent oracle-diff workflow + the stable production scheme (`B=KE+g(h+h_s)`, operators_cdgrid.py
  :1458) independently CONFIRM FV3 SW D-grid momentum IS geopotential-forced. FB-only (no prod impact).
- **BUG 2 (config; the growth rate): the FB chain needs the DUOGRID (FV3 operational ng=3).** The
  inviscid growing mode is VERTEX-seeded (vertex-max 2-3× interior, argmax at cube corners). The
  non-duogrid `_corner_vorticity` (fv3_sw_core.py:1234-1254) uses `mode='edge'`/linear-extrap halos
  + applies the FV3 corner-correction (sw_core.F90:397-400) with the WRONG (edge-extrap, not
  cross-face) fy → spurious vertex vorticity. With `create_cubed_sphere(use_duogrid=True,
  duogrid_ng=3)` (cross-face halos), FB+Phase4 W2 C36 survives DAY 1 (was NaN@47≈4h) → NaN ~33.6h =
  **8.5× stability gain**. Earlier "FB unstable" tests used the WRONG (non-duogrid) config.
- **RESIDUAL (open):** day1 max|u_d|=50 (vs 38), W2 err 2.5e-2, day2 NaN. Slow residual growth.
  Candidates: duogrid `_corner_vorticity` dxc/dyc edge-mode METRIC halo (lines 1209-1215, "O(dx)");
  Phase4 a2b in the duogrid context; the d2a2c duogrid path. NEXT: drive the residual down. iter79: t=0 single-step |dh| is NOT a good predictor — duogrid has LARGER t=0 dh (rms 7.13 vs
  non-dg 2.13, v/i 5.7) yet SLOWER growth (more stable), so the residual is a GROWTH-RATE mode
  (needs multi-step / eigen analysis), not a t=0-amplitude one. FB chain: 2 high-value fixes
  extracted (PGF + duogrid, 8.5×); residual is a deep long-tail — diversify before grinding more.
  iter81: Phase4 a2b RULED OUT as the residual cause — `_pad_halo_auto_h2` (used by
  `_interp_center_to_corner_a2b_ord4`) IS duogrid-aware (passes `duogrid=dg`), so the Phase4 corner
  geopotential uses the cross-face halo, consistent with the rest. Residual narrowed to the duogrid
  `_corner_vorticity` dxc/dyc edge-mode METRIC halo (O(dx), authors judged minor) or the d2a2c
  duogrid path / c_sw coupling subtlety — a deep eigen-analysis task, not a single-iter fix. iter82: DISSIPATION RULED OUT as the residual fix — FB+Phase4+duogrid W2 C36 dissipation sweep:
  FV3-default day2-NaN; 2× del4+Smag NaN@16 (WORSE — heavy del4 hits its own explicit-stability
  limit at dt=300); nord=2 del6 no help. More dissipation = worse, both before & after Phase4+
  duogrid. ⇒ the residual is a STRUCTURAL corner coupling/metric bug (vertex-seeded growth mode),
  NOT a tunable grid-scale dissipation mode. Remaining candidate = the duogrid `_corner_vorticity`
  metric halo / d2a2c-duogrid / c_sw corner coupling — a deep structural fix vs the oracle.

## gnomonic_ed GRID (FV3 operational gt=0; iter62-78) — WIRED + clean + codex-approved
create defaulted to EQUIANGULAR (gt=2, aspect 1.40); FV3 uses gnomonic_ed (gt=0, aspect 1.06, dx √2)
to suppress high-res corner/edge modes. **DELIVERED:** `create_cubed_sphere(gnomonic="ed")` (A-grid)
+ `create_cubed_sphere_cdgrid(gnomonic="auto")` (infers from base aspect: ed<1.15, eq>1.25). Both
equiangular byte-identical (cube SW 4/4 unchanged). Grid = `_face_gnomonic_to_lonlat(f, meshgrid(
ed_angle_1d))` — separable per face ~1e-16, reproduces FV3 native corners to 3e-15. Codex APPROVED.
- **iter73 halo-collapse (FIXED):** the construct-extended halo collapsed at the 4 cube corners
  (zero-width = duplicate nodes; latent). Fix: build the ed extended grids (supergrid/corner_ext/
  padded_supergrid) via `_gnomonic_ed_faces_from_angle_1d` + 1D `_gnomonic_ed_extrap1d` (reduces
  EXACTLY to equiangular linspace for a uniform array). Tests 29/29 + `test_gnomonic_ed_halo_
  nocollapse_iter73.py` 4/4. (Ruled out neighbor-fill — injects the cube-edge crease, regressed.)
- **EIGENMODE: gnomonic_ed PARTIALLY fixes C96 W5** (same eq-tuned damping): eq daily max|u_d|
  26/32/33/40/80 (blows) vs ed 26/32/33/33/48/55/103/104/87/83 (bounded to 10d, saturates not NaN).
  Delays/bounds but doesn't eliminate; residual needs the FB chain or ed damping recalibration.
- **OPEN — ed 8× W2 corner error (a real, non-convergent IN-DOMAIN bug, NOT the halo):** W2 C36 ed
  ss-err 1.4e-3 vs eq 1.76e-4. t=0 |dh|max NON-CONVERGENT (ed C24/36/48 = 1.77/1.80/1.79 STAGNANT vs
  eq 0.073/0.051/0.039 converging) ⇒ O(1) inconsistency at the cube corner. RULED OUT: IC (canonical),
  area (matches spherical-excess 5.5e-8), edge-angles (exact from corners), grid build, A/cd mismatch
  (bit-identical), d2a2c halo (corner-cell dh uses in-domain ops not halo). ⇒ the residual is the
  IN-DOMAIN corner-cell C-grid contravariant transform (cos_sg5/rsin2) + PPM flux at the corner =
  the SW-core corner treatment (convergent with the FB-chain work). Micro-investigation STOPPED.

## VERIFICATION
- SW 16/16 all grids ≤1 cell. Full fast cube atm dynamical suite PASS + NH dcmip_tc1, mass machine-
  zero. Ocean matrix cube 9/9; rest_state ×12 machine-zero.
- Cross-grid CLOSENESS (W2 height L2 vs exact): cube 1.76e-4 BETWEEN ico 9.9e-5 and latlon 2.67e-4.
  Regression-clean 44/44.
- **iter80 re-confirm at HEAD (post gnomonic_ed/FB commits — NO regression):** cube SW matrix
  4/4 PASS — W2 L2=1.76e-4 (unchanged), v_ll_Linf=0.339, W5/W6 mass drift ~1e-15, cosine_bell
  L2=0.131. Visual edge-artifact PNGs (W2 v-wind + W5 wind_speed, C36 & C96) SURFACED to the
  user for the directive's human visual inspection (assistant barred from Read images).

## ACTION QUEUE
[TODO bounded] human visual PNG inspection (W2 v / W5 wind_speed surfaced iter80). (a2b_ord4
regression: DONE — covered by test_a2b_ord4_linearity_iter302 / _theta_corner_iter700 / interp_center_iter306.)
FOUNDATIONAL (per user no-improvise, use oracle): FB chain residual (drive duogrid+Phase4 W2 to
stable); ed in-domain corner SW-core treatment; 3D PE upwind PPM vort; thread FV3 sin_sg(5) seam
rotation; C96 W5 eigenmode (converges on the FB-chain/SW-core work). Climate cross-grid CPU-INFEASIBLE.
State persisted: memory `cube-fv3-faithfulness-state`.
