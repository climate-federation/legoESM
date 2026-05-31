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
(`fv3_fb_sw_step` / `FV3FBShallowWaterModel` exists but unstable). 3D PE D-grid vort/KE = centered
`zeta_corner*v_d` (primitive_eq_cdgrid:445), NOT FV3 upwind PPM `hord_vt`.
**iter76 — FB instability REFRAMED (it's a COUPLING BUG, not tuning):** W2 C36 equiangular blows
to NaN at a dt-INDEPENDENT physical time ≈3 h — dt=300→step43(3.6h), dt=200→step59(3.3h),
dt=100→step108(3.0h). dt=300 ZERO-dissip NaNs at the SAME step 43 as FV3-dissip; HEAVY dissip
(div_damp=8,d4_bg=0.5) NaNs FASTER (step 11). ⇒ a continuous-time GROWING MODE, NOT a CFL/dt
instability, and the del4/Smagorinsky dissipation does NOT damp it (heavy dissip adds its own
explicit-stability blow-up on top). Production RK3 is stable on the SAME W2/grid ⇒ the unstable
mode is SPECIFIC to the FB c_sw→d_sw coupling = a BUG vs the FV3 oracle (sw_core.F90 `c_sw`/
`d_sw`), NOT a dissipation-tuning or timestep problem (FV3 runs FB stably). The long-standing
"FB unstable, needs interface dissipation" framing was WRONG.
**iter76 — ROOT CAUSE FOUND (oracle-diff workflow + self-confirmed): the FB step OMITS the
D-grid backward pressure-gradient phase.** `fv3_fb_sw_step` (fv3_sw_core.py:1946) has only:
Phase1 `_c_sw` (dt/2) → Phase2 `_p_grad_c` (adds the PGF dp_x/dp_y to **uc/vc ONLY**) → Phase3
`_d_sw_native` (u_d_new = u_d + (ke_diff + fy_vort)·rdx). The prognostic D-grid winds u_d/v_d
NEVER receive the pressure-gradient force — FV3's Phase4 `one_grad_p`/`grad1_p_update`
(dyn_core.F90:2347/2483, the D-grid backward PGF using the a2b-corner geopotential) is ENTIRELY
MISSING. For W2 geostrophic balance (Coriolis ↔ PGF) the prognostic winds have NO PGF restoring
⇒ the balance is unbalanced ⇒ the dt-independent ~3 h growing mode. (Spatial check: by
saturation the mode is grid-scale/~global with a weak corner seed — consistent with a global
missing-restoring-force mode, not an edge-only artifact.) The port HAS `_p_grad_c` (C-grid) but
NO D-grid PGF; production RK3 uses an Arakawa-Lamb D-grid PGF that works. FIX = add Phase4 to
fv3_fb_sw_step: the FV3-faithful `one_grad_p` D-grid backward PGF on u_d/v_d (geopotential
g·(h_new+h_s) → a2b to corners → gradient along D-grid edges → u_d += -∂x(gz)·dt etc.), backward
time-centered on the post-d_sw height. Validate: FB W2 C36 no longer blows at ~3 h. (codex/oracle
review the new phase.)

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
**EIGENMODE: gnomonic_ed PARTIALLY fixes it (iter71, full 10-day picture — verify-first
corrected an optimistic day-5 snapshot):** W5 C96, SAME equiangular-tuned damping —
equiangular daily max|u_d| 26/32/33/40/80 (steep blow-up) vs gnomonic_ed
26/32/33/33/48/55/103/104/87/83 (to 10d). gnomonic_ed DELAYS + bounds the eigenmode (day5
48 vs 80; SATURATES/recovers to 83 rather than NaN-ing) but does NOT eliminate it — it still
grows to ~104 by day 7. ⇒ the C96 W5 eigenmode is PARTLY grid-conditioning (ed clearly
better: slower onset, bounded vs blow-up) but has a RESIDUAL component the grid alone doesn't
fix with equiangular-tuned damping. Full fix needs ed-specific damping recalibration AND/OR
the core c_sw/d_sw faithfulness (centered-scheme contribution). Both A-grid + C/D ed grids
build (gnomonic="ed"; ed cdgrid dxc √2, sin_sg∈[0.866,1]).
**CODEX iter72 (high, FIXED):** caught a corruption path — ed A-grid + model constructor
(calls `create_cubed_sphere_cdgrid(grid)` no-flag) would silently get an equiangular C/D
grid (mismatched metrics). CubedSphereGrid is a JAX-pytree NamedTuple (string gnomonic field
breaks JIT), so `create_cubed_sphere_cdgrid` now defaults `gnomonic="auto"` and INFERS from
base cell-aspect (ed<1.15, equiangular>1.25, ambiguous raises). Model constructors auto-get
matching C/D metrics. Equiangular byte-identical. Test: auto-on-ed==explicit-ed (5/5).
Codex re-review APPROVED: corruption path closed, ed θ layouts verified line up with the
supergrid/corner-ext/padded-supergrid indexing, no material findings. ⇒ gnomonic_ed grid
wiring (A-grid + cdgrid + auto-infer) COMPLETE + codex-approved + corruption-safe.

## ⚠️ iter73 — ed cdgrid CORNER-METRIC BUG (verify-first; corrects the "sound" claim)
Codex verified θ LAYOUTS, NOT metric VALUES at the corner. W2 C36 1-day on the ed grid:
DYNAMICS CORRECT (mass drift 1e-8, max|u_d| 38.7 ✓) but steady-state h-error **1.4e-3 vs
equiangular 1.76e-4 = 8× WORSE**. Damping-INDEPENDENT (1.40→1.30→1.23e-3 over a 4× sweep) ⇒
NOT tuning. Localized: t=0 |dh| imbalance (zero damping) max **1.79 @ face1 (i=0,j=0) = a CUBE
CORNER**, 35× the equiangular 0.051; the 4 corners spike (~1.79) while edge-MID (1.08e-2) and
deep interior (8.9e-4) are HEALTHY (~equiangular). The cube corners are the smallest, most
non-orthogonal ed cells (cosa ~ -0.5).
COLLAPSE BUG (real, latent, NOW FIXED): the ed extended grids fed halo-extended theta to
`_gnomonic_ed_construct`, which hardwires the W/E edges to the boundary meridians (where the
gnomonic coord pp2 = -tan(0.75pi)*rsq3 is CONSTANT for all lat) -> no edge-PERPENDICULAR halo
row -> the corner halo nodes COLLAPSE onto the boundary (zero-width = duplicate grid nodes).
In-domain matched grid.lon/lat to 1.4e-15 (why A-grid tests + codex layout-review passed). FIX:
the ed grid is EXACTLY `_face_gnomonic_to_lonlat(f, meshgrid(ed_angle_1d))` (separable per face
~1e-16, 6 faces congruent, reproduces FV3 native corners to ~3e-15). So
`gnomonic_ed_{supergrid,corner_ext,padded_supergrid}_lonlat` now build via
`_gnomonic_ed_faces_from_angle_1d` with the non-uniform ed 1D angle array (`_gnomonic_ed_angle_1d`
from the native grid) extrapolated +-1 node (`_gnomonic_ed_extrap1d`, linear -> reduces EXACTLY
to equiangular linspace for a uniform array). Halo no longer collapses; interior bit-exact vs
corners/supergrid; consistent. Tests 29/29 + new `test_gnomonic_ed_halo_nocollapse_iter73.py` 4/4.
RULED OUT: neighbor-fill (FV3 fill_corners via CONNECTIVITY) REGRESSED to imbalance 58 (neighbor
node ~1 cell off the same-face continuation -> injects the cube-edge CREASE into the centred-diff
tangent; the cdgrid wants the SAME-FACE continuation); full-range mirror + drop-overrides also
collapse.
**BUT W2 is UNCHANGED by the halo fix -- bit-identical 1.795 / 1.4e-3 before & after.** So the
collapse was a real LATENT bug (would bite higher-halo / future halo consumers) but is NOT the
W2 cause: cosa_corner normalizes the tangent -> insensitive to halo node position. ed-vs-eq
CORNER metrics are CLOSE (cosa_corner both -0.5; cos_angle_edge_x 0.871 vs 0.872; dxc 2.19e5 vs
2.57e5 = genuine grid diff). => the 8x W2 error is an IN-DOMAIN property of the ed corner cells
(smallest + most non-orthogonal) x the FV3Edge SW core -- NOT a gross metric bug, NOT the halo.
STILL OPEN: deeper SW-core / non-orthogonality analysis (or possibly inherent to ed+this scheme;
cross-check vs the FV3 oracle's own W2-on-ed corner residual).
STATE: ed builds, stable, mass-conserving, halo CLEAN; the 8x W2 corner error is the open defect.
Prior "cdgrid COMPLETE + sound" still premature on W2 accuracy.
CODEX adversarial-review (commit ffad96fd): APPROVE, no material findings — confirmed the
separable 1D angle reconstruction matches native gnomonic_ed on all 6 faces to roundoff,
refinement-consistent for even AND odd n, equiangular branch untouched.
CONVERGENCE TEST (iter74) — the 8x W2 error is a BUG, not inherent: t=0 |dh|max (zero damping)
equiangular C24/36/48 = 0.073/0.051/0.039 (CONVERGES ~1st order) vs ed = 1.766/1.795/1.788
(STAGNANT, rate ~0). A consistent scheme's truncation must vanish with resolution; ed's corner
max is PINNED at ~1.79 at all resolutions (while rms slowly converges 0.138/0.104/0.086) ⇒ a
LOCALIZED O(1) inconsistency at the cube corner (3-face junction), not inherent ed truncation.
RULED OUT: A-grid/cdgrid construction mismatch (the imbalance was BIT-IDENTICAL before/after the
cdgrid rebuild from construct→`_face_gnomonic_to_lonlat`; construct and the new path agree to
3e-15). REMAINING HYPOTHESES for next iter: (1) discrete metric-identity / freestream-
preservation violation at the ed corner (A-grid `grid.area` from `_compute_exact_cell_areas_ed`
vs cdgrid face-lengths dxc/dyc don't close the divergence stencil); (2) the FV3Edge SW-core
3-face-junction corner special-casing (sw/se/nw/ne_corner) calibrated for equiangular's specific
corner angle, not ed's.
DISENTANGLED (iter74) — it is a REAL MODEL bug, not the harness: the IC projection
`u_d=cos_angle_edge_x·u0·cos(lat_edge_x)` IS the canonical one (verbatim in the production W2
runner run_w2_w5_cosine_bell_iter1030.py:77). RULED OUT as the cause: area (ed grid.area matches
spherical-excess of corners to 5.5e-8, corner cell 1.3e-8), edge-angle metrics (lat_edge_x/
angle_edge_x computed from EXACT in-domain corner geometry, no halo dependence), grid
construction (clean, codex-approved), A-grid/cdgrid mismatch (bit-identical). Freestream (h≡const)
shows BOTH grids have O(1) velocity-flux divergence (eq 0.82, ed 3.10) — eq's full-W2 cancels it
to 0.039 via the h-flux term (discrete steady balance holds) but ed only to 1.79 (balance fails).
PARADOX: ed is MORE uniform (aspect 1.06 vs eq 1.40) yet does WORSE ⇒ not cell distortion.
NARROWED ROOT: the C-grid CONTRAVARIANT TRANSFORM consistency — the IC/wind uses the edge-tangent
angle (cos_angle_edge, i-tangent→east) but the flux divergence transforms covariant→contravariant
via cosa_u/sina_u/rsin (from sin_sg sub-grid angles); these two angle systems must be mutually
consistent for the discrete W2 balance to cancel. Consistent for eq's separable parametric map,
apparently NOT for ed. STRATEGIC: this + the C96 W5 eigenmode BOTH converge on SW-CORE
FAITHFULNESS (FV3 c_sw/d_sw flux treatment) — the next big faithful effort.
iter75 — d2a2c HALO ruled out too: passing the non-orthogonal seam rotation (cos_theta/sin_theta
= cos_sg5/sin_sg5) to the d2a2c `pad_halo_vector` (fv3_sw_core.py:602) left the W2 imbalance
BIT-IDENTICAL (eq 0.051, ed 1.795). Reason: the t=0 dh at the literal CORNER CELL (i=0,j=0) uses
its IN-DOMAIN contravariant transform (step-3 cos_sg5/rsin2_cell) + in-domain C-grid fluxes — the
cross-seam halo never reaches the corner-cell dh. ⇒ the ed corner inconsistency is in the
IN-DOMAIN corner-cell C-grid operators (the step-3 covariant→contravariant transform cos_sg5/
rsin2 + the PPM flux + dxc/dyc at the corner), NOT any halo. Experiment reverted (inert no-op).
STOPPING the ed-W2 micro-investigation (diminishing returns; thoroughly ruled out IC, area,
edge-angles, grid build, A/cd mismatch, d2a2c halo — the residual fix IS the in-domain SW-core
corner treatment = the convergent SW-core-faithfulness work, to be done vs the FV3 oracle's
c_sw/d_sw, NOT micro-probed further). ed grid remains USABLE (stable, conservative, halo clean)
with a known 8× W2 corner-accuracy gap vs equiangular.
**(prior scoping, now DONE) the CDGRID layer:** dynamics consume
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
