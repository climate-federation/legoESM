# FV3-faithful cubed-sphere — change log (shrunk @ iter ~70)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`,
70 F90), matching MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge
artifacts**, visual+quantitative, codex/oracle-reviewed. CPU only (Metal broken).
**User directive: never A-grid; be FV3-faithful; use the Fortran as oracle; do NOT
improvise.** Branch `latlon-fv-amip-verify`, ~110 commits ahead. Per-iter detail in git.

## ✅ FIXES (done, validated, codex-reviewed)
- ATM panel-edge imprint: cc→D-grid lift (u,v) as scalars (7 sites). baroclinic v_rms
  4.95→0.60, mass→1e-15. Test `test_vector_cc_to_dgrid_wind_lift.py`.
- CROSS-GRID alignment: canonical `_canvas_lat/_canvas_lon`. Test `test_latlon_regrid_alignment.py`.
- OCEAN geostrophic 40%→1.45%: cube barotropic A-grid → FV3 SW core `fv3sw` (never A-grid).
  + FC viscosity vector-halo. Faithful upwind option `fv3edge` (gated, 8/9).
- [iter57] stale iter-707 KE-fingerprint test → rtol/L2-scaled. [iter62] `hord==9` iv
  mislabel → `_pert_ppm_iv0` (FV3 iord=9=iv=0; unexercised; cube SW 4/4 unchanged).

## EDGE-ARTIFACT TRUTH (iter57-61; earlier "PROVEN clean" was OVERSTATED)
Reliable metric = `compute_cross_face_continuity` (halo.py): cross-seam/interior first-diff
via real `pad_halo_4d` (≈1 continuous, 5-50× = real jump). SUPERSEDES pooled same-face
`compute_edge_artifact_metric` (unreliable; gave 5.95× on W5 v where true continuity is 1.30×).
Codex caught+fixed a denominator bug; re-APPROVED. Tests 4/4.
- BAROCLINIC (3D PE): genuinely clean, converges C36→C72 (≤1.24× incl rotated).
- SHALLOW-WATER: all PHYSICAL fields CONTINUOUS — C36/C48/C96 FLAT. No large edge artifact.
- **OPEN — C96 W5 edge eigenmode:** max|wind| tracks C36/C48 (~38) to day3.5 then BLOWS UP
  38→79 (edge/corner faces 0,4). CFL REFUTED (dt-independent). NOT damping (dddmp inert in
  prod RK3) and NOT flux (mass conserved). Same mode in the FB chain (mass-conserving h-drift
  at i=0 edge). Every exposed knob inert. ⇒ root = the GRID (equiangular corner distortion)
  + core c_sw/d_sw. FAITHFUL fix = gnomonic_ed grid (likely) — being wired; see below.
- STILL MISSING: human visual PNG inspection (assistant barred from Read images):
  `results/atmosphere/shallow_water/williamson{2,5}/cubed_sphere/C36/snapshots_{v,wind_speed}_native.png`.

## FAITHFULNESS SCORECARD (12-agent oracle workflow iter57; Fortran-vs-port direct read)
**FAITHFUL but ORPHANED:** gnomonic_ed grid (1 ULP — NOW being wired, below); get_area/
cell_center3 (prod uses equiv l'Huilier); upwind vort-flux/`fv_tp_2d` (3D PE doesn't call it).
**MINOR:** `d2a2c_vect` (live duogrid path faithful); `a2b_ord4` (inert default); cross-face
vector halo `pad_halo_vector_4d` ORTHOGONAL — drops FV3 `1/sin_sg(5)` → O(cosθ) seam err;
faithful `pad_halo_dgrid_vector_4d` (12/12) gated off.
**MAJOR:** SW core prod = single-stage SSP-RK3 (1 Arakawa-Lamb tendency), NOT FV3 FB c_sw→d_sw
(`fv3_fb_sw_step` exists but unstable/unreachable). 3D PE D-grid vort/KE = centered
`zeta_corner*v_d` (primitive_eq_cdgrid:445), NOT FV3 upwind PPM `hord_vt`.

## GRID gap → gnomonic_ed wiring (LIKELY fixes the C96 eigenmode; iter62-69)
create defaulted to EQUIANGULAR (gt=2, corner aspect 1.40); FV3 operational = gnomonic_ed
(gt=0, aspect 1.06, dx √2). FV3 uses gnomonic_ed to suppress high-res corner/edge modes ⇒
wiring it likely fixes the C96 eigenmode.
**✅ A-GRID GATE DELIVERED (iter68):** `create_cubed_sphere(gnomonic="ed")` builds the FV3
gnomonic_ed CubedSphereGrid in production. Bricks (all verify-first, equiangular default
BYTE-IDENTICAL — cube SW 4/4 unchanged, W2 v_ll 0.339):
  - `_gnomonic_ed_construct(theta_w, alpha)` halo-extensible core (diagonal pinned to ±α);
  - `_gnomonic_ed_remap_to_create` (perm [0,1,3,4,5,2] + D4 rot [0,0,1,1,0,3]) — solved the
    FV3↔create face-numbering + SEAM-CONTINUITY crux (validated cross-face 1.22);
  - centers = `cell_center2` of corners (defn A; construct-at-cell-centre-θ is range-dependent/
    WRONG); `compute_padded_{half_metrics,angle}_ed` + `_compute_exact_cell_areas_ed` all defn-A
    (corner-based via `_gnomonic_ed_padded_centers`, in-domain matches grid.lon/lat to 1.4e-15);
  - `compute_halo_interp_offsets_ed`/_hN (h1/2/3) via position-matching on the cleanly-extending
    centres (codex's gating blocker; bounded maxabs 0.223).
  ed grid: area 4πR², dx √2, aspect 1.05, rejects Schmidt/shift. 7 verify-first subtleties
  caught (separable-dist, moved-diagonal, face-perm, seam, halo-saturation, range-dependence,
  centre-consistency); codex-reviewed builders SOUND. Tests: gnomonic_ed bricks 30+/30+.
**⏳ NEXT MAJOR LAYER — the CDGRID (gates the eigenmode test):** dynamics consume
`create_cubed_sphere_cdgrid(base)`, which REBUILDS its own equiangular C/D supergrid
(`_compute_supergrid_metrics(n, _face_gnomonic_to_lonlat,…)` + `linspace` α at cubed_sphere_
cdgrid.py:203/315/579/641/785/952/960), ignoring base's distribution. ed cdgrid = a separate
~1200-line equiangular rework (supergrid area/dxc/dyc, corners, edge/corner angles, sin_sg,
corner-gradient matrix, extended grids); built on a per-face PARAMETRIC map (equiangular)
which ed (non-separable, mirror-based) doesn't fit → needs restructure around precomputed ed
6-face grids. Then dynamical-suite validation + damping re-calibration + W5 C96 eigenmode test
(the payoff). **A-grid gate = DELIVERED; cdgrid + dynamics = remaining major effort; eigenmode-
fix hypothesis UNCONFIRMED until then.**

## VERIFICATION
- SW 16/16 all grids ≤1 cell. Full fast cube atm dynamical suite PASS + NH dcmip_tc1, mass
  machine-zero. Ocean matrix cube 9/9; rest_state ×12 machine-zero.
- Cross-grid CLOSENESS (W2 height L2 vs exact): cube 1.76e-4 BETWEEN ico 9.9e-5 and latlon
  2.67e-4 (cube > latlon). Regression-clean 44/44. Health 23/24 (1 = collaborator's latlon
  `implicit_cn`, not cube scope).

## ACTION QUEUE
SAFE-BOUNDED: [DONE] hord==9, cross-face denominator bug, gnomonic_ed signature test.
[TODO] a2b_ord4 numerical regression; human visual PNG inspection.
FOUNDATIONAL (multi-day; per user no-improvise): cdgrid ed rework → run W5 C96 eigenmode test
(IN PROGRESS — A-grid gate done); stabilize+wire FB chain (FV3 c_sw/d_sw 2-stage + dddmp);
3D PE upwind PPM vort; thread FV3 sin_sg(5) seam rotation; climate cross-grid CPU-INFEASIBLE.
State persisted: memory `cube-fv3-faithfulness-state`.
