# FV3-faithful cubed-sphere — change log (shrunk @ iter ~43)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`,
readable, 70 F90), matching MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube
edge artifacts**, visual+quantitative, codex-reviewed. CPU only (Metal broken).
**User directive: never A-grid; be FV3-faithful.** Branch `latlon-fv-amip-verify`,
~63 commits ahead of main. Per-iter detail in git.

## ✅ FIXES (done, validated, codex-reviewed)
- **ATM panel-edge imprint**: cc→D-grid lifted (u,v) as scalars (no cross-face
  rotation) → 7 sites use `center_to_dgrid_vector`/`pad_halo_vector_4d`
  (ac6a8f58…4fd71710). baroclinic v_rms 4.95→0.60, converges, mass→1e-15.
  Test `test_vector_cc_to_dgrid_wind_lift.py`.
- **CROSS-GRID alignment**: latlon/gaussian stored native width under 360-pt label
  (tens-of-deg translation) + node-vs-cell-centered drift → one canonical canvas
  `_canvas_lat/_canvas_lon` for all regrid+npz+axes (2d04a168/882b4063/418cc7de).
  Also unblocked latlon SW (was 0%-running: bad import). Test
  `test_latlon_regrid_alignment.py`.
- **OCEAN geostrophic non-zonal artifact** (40%→1.5%): cube barotropic was A-grid
  (computational pressure mode). Routed through validated FV3 cube SW core
  `CDGridShallowWaterModel.step` → `barotropic_substeps_fv3sw`,
  `barotropic_staggering="fv3sw"` (bd74c45b). `h=H_bathy+eta`, `h_s=-H_bathy`;
  ocean-tuned `barotropic_sw_div_damp_factor=120`. Land: SW `fix_mass=False` +
  per-substep corner-ocean wind mask (coasts impermeable, land eta=0, rest
  machine-zero). Solution-space: a_grid=40%, explicit-f*v c_grid=NaN, bare-VI
  FE/RK3=NaN; only full SW core stable+zonal. Test `test_fv3sw_barotropic.py`.
  RE-VERIFIED iter ~55 (fresh): cube geostrophic PASS, T drift 7.5e-14,
  max_speed 0.0143≈latlon/mpas, non-zonal fraction 1.45% (A-grid was ~40%).
- **OCEAN FC velocity viscosity** vector-halo (4fe7108c).

## EDGE-ARTIFACT STATUS (CORRECTED iter ~57 by 12-agent oracle+adversarial workflow)
**Honest verdict: NOT-YET-ESTABLISHED as a general claim. The earlier "edge-clean
PROVEN" was OVERSTATED for shallow-water.**
- BAROCLINIC (3D PE): genuinely clean — uniform per-face roughness, magnitude-
  normalized roughness SHRINKS with resolution (C36→C72 in results/probe_conv_fixed),
  converges. SUPPORTED-WITH-CAVEATS. (Fresh iter56: cube 3/3 PASS, mass 7e-15..3e-13,
  v-wind roughness ≤1.24× incl rotated_baroclinic crossing seams.)
- SHALLOW-WATER: REAL coherent cube-vertex/equatorial-seam imprint in v-wind /
  wind_speed that my POOLED same-face metric HID. 4 adversarial skeptics all REFUTED
  the clean claim: (1) same-face band=2 metric excludes the seam row + has a high
  detection floor; (2) the committed `compute_edge_artifact_metric` (halo.py:2994) is
  pooled same-face-ONLY (no cross-face term) and the fresh iter55/56 reverify only ran
  metric-A, never re-ran cross-face metric-B on fresh data; (3) numbers-only violates
  CLAUDE.md "visual REQUIRED" — native PNGs never opened; (4) pooling hides per-face
  amplification + polar "clean" is small/large MASKING. Independently re-verified:
  W2 v_cc_north 16/24 corner cells show 2nd-diff >2× median (same-face metric).
**RESOLVED iter ~58 via the CORRECT metric (TRUE cross-face continuity through the
model's real `pad_halo_4d`: cross-seam first-diff RMS / same-face interior first-diff
RMS — a real discontinuity = 5-50×):**
- W5 (PHYSICAL flow, 5-day, mass 4.9e-16): v_cc_north (max 20.2) cross-seam **1.30×**;
  wind_speed 1.58×; height 2.19× (one face). CONTINUOUS.
- W2: physical u 1.57×, height 1.62×; v (ERROR field, max 0.34) 2.23×. CONTINUOUS.
⇒ BOTH overclaims corrected. The skeptic's "W5 v 5.95×" used the unreliable SAME-FACE
2nd-diff metric (amplifies edge curvature); true cross-face continuity is 1.30× for
physical W5 v. My earlier "PROVEN clean" was loose; the skeptic's "real coherent
imprint" was also overstated. TRUTH: NO large edge artifact (all physical SW fields
continuous ≤2.2×); the same-face roughness metric is UNRELIABLE in BOTH directions and
should not anchor the claim. Mild error-field elevation (W2 v 2.23×, sub-0.34 m/s
residual) is the tiny discretization residual, invisible to L2/mass. Baroclinic clean.
- [DONE iter58] TRUE cross-face metric committed: `compute_cross_face_continuity`
  (halo.py) + `tests/grids/test_cross_face_continuity_metric.py` (3/3: smooth field
  <3×, injected seam jump >10×, 3D/4D shapes). Supersedes the unreliable same-face one.
- [DONE iter59] C36/C48/C96 cross-face continuity sweep (5-day, my new metric):
  W2 FLAT across res (u 1.57→1.56→1.55, height 1.62→1.60→1.61) — CONTINUOUS, no
  res-growth. **But W5 (mountain flow) FOUND a real C96 EDGE EIGENMODE:** max|wind|
  tracks C36/C48 (~38 m/s) until day 3.5 then BLOWS UP 38→51→78.9→62 (days 4-5),
  edge-localized (face0 peak 78.9 at (14,94)=edge, face4 76.8 at corner); wind_speed
  cross-seam ratio 1.58(C36)→3.07(C96). C36/C48 stay flat → genuine C96-only cube
  edge eigenmode (matches the runner's documented iter-65/69/79 "C96 eigenmode" that
  dt-tuning has been chasing — dt=150 NaNs day15, dt=100 NaNs day22.5).
  **Per user directive (FV3 oracle, NO improvising): do NOT band-aid with dt/damping
  tuning.** Root cause traces to the oracle-identified non-faithful pieces: equiangular
  grid (worse corner distortion vs FV3 gnomonic_ed √2), orthogonal seam rotation (drops
  FV3 sin_sg(5)), single-stage RK3 + non-FV3 del2/del4+dddmp damping (FV3's divergence
  damping is tuned to kill these grid-scale edge modes). Faithful fix = foundational
  FV3 work, NOT dt hacks. [oracle damping comparison = next]
- STILL MISSING: visual PNG inspection (human); FV3-oracle SW damping comparison
  (nord/d2_bg/d4_bg/dddmp) to ground the C96 eigenmode root-cause.

## VERIFICATION
- SW 16/16 all grids, aligned within 1 cell. Full fast cube atm dynamical suite
  PASS (baroclinic, rotated_baroclinic, gravity_wave, rossby_haurwitz,
  mountain_rossby, inertio_gravity, rotated_steady, rest_state_topo, dcmip_11/12)
  + NH dcmip_tc1, mass machine-zero. Ocean matrix cube 9/9; rest_state ×12
  machine-zero all grids.
- Cross-grid CLOSENESS (fresh iter ~56, W2 height L2 vs exact): cube 1.76e-4
  BETWEEN ico 9.9e-5 and latlon 2.67e-4 — cube MORE accurate than latlon, near
  ico, squarely in FV-family band (spectral 3.6e-8 = exponential, diff class).
- Cross-grid CLOSENESS: gravity_wave max|v| cube 19.3≈latlon 20.2 (5%);
  geostrophic max_speed 0.0142≈latlon/mpas; phillips 0.44≈latlon 0.41;
  barotropic_wave cube least-dissipative (most accurate). Regression-clean 44/44.
- Health re-sweep 23/24 (my modules clean incl `test_barotropic_cgrid`). 1 FAIL =
  latlon `implicit_cn` solver (collaborator Dhruv Balwada's Mercator/conservation
  work, NOT cube scope, memorialized).

## GRID faithfulness (oracle, iter ~52-53)
legoESM cube grid is **EQUIANGULAR** gnomonic (alpha equal-angle in [-π/4,π/4],
x=tan(alpha) — FV3 `gnomonic_angl`/grid_type=2; cubed_sphere.py:618/659). FV3's
**operational** grid is `gnomonic_ed` (grid_type=0, fv_grid_utils.F90:1313) — a
DIFFERENT, more cell-uniform variant. Same family + same spherical-excess areas
(total 4πR² to 5e-7). Docstring fixed (wrongly said "equidistant").
**FV3 gnomonic_ed grid ALREADY FULLY PORTED + TESTED (iter ~54):** `gnomonic_ed`
(cubed_sphere.py:1833, faithful FV3 port), `gnomonic_grids` dispatcher (:1786),
`mirror_grid_faces`, and the full 6-face builder **`make_fv3_native_grid(im,
grid_type=0)`** (:1745) — all tested (test_fv3_gnomonic_ed/grids/mirror/native
_iter621-629). Verified: 6-face (6,n+1,n+1), great-circle dx max/min = **1.4130 ≈
√2** (exact FV3 signature). So the FV3 grid math is DONE + correct. **The gap is
purely OPERATIONAL: `create_cubed_sphere` uses the equiangular path
(_compute_gnomonic_lonlat), NOT `make_fv3_native_grid`.** Wiring is non-trivial: `create_cubed_sphere` derives ALL metrics + angles from the
equiangular gnomonic centers + a PADDED gnomonic halo extension (cubed_sphere.py:
280,319) — so gnomonic_ed needs the full metric + halo-extension re-derivation on
that grid (not a corner-swap) + re-validate ALL dynamics (calibrated on
equiangular). Sanctioned effort; grid MATH done+tested, but the metric/halo
construction + re-calibration remain. Real FV3-grid-variant gap.

## FAITHFULNESS SCORECARD (12-agent oracle workflow, iter ~57; each verdict from
## direct Fortran-vs-port read; synthesis cross-checked load-bearing claims)
**FAITHFUL ports — but ORPHANED (NOT on the running model):**
- `gnomonic_ed`/`gnomonic_grids`/`symm_ed`: **bit-faithful** (agent reimplemented FV3
  gnomonic_ed in numpy independently → max|dlon/dlat| = 4.4e-16, 1 ULP). But ZERO
  production callers — `create_cubed_sphere` runs EQUIANGULAR (grid_type=2). [grid gap]
- `get_area`+`cell_center3`+`spherical_angle`: faithful; production `grid.area` uses an
  equivalent l'Huilier split (agrees rel<1e-6) — second algorithm, not FV3's stencil.
- upwind abs-vorticity flux / `fv_tp_2d` / xppm-yppm: line-for-line faithful (iord=9
  limiter) — but ORPHANED (3D PE doesn't call it). Latent `hord==9` iv-mislabel
  (fv_tp_2d.py:504, iv=1 where FV3 iord=9 is iv=0; unexercised).
**MINOR divergence (partially wired):**
- `d2a2c_vect`: LIVE duogrid 4th-order path faithful (FV3 also gates edge/corner
  specials off under `dg%is_initialized`); non-duogrid corner sign-flip overrides
  dropped (off live path). Live only in SW FV3EdgeShallowWaterModel + opt-in ocean.
- `a2b_ord4` zeta_corner: interior PPM+Lagrange bit-faithful; INERT by default
  (`use_fv3_a2b_zeta_corner=False`), LIVE only under `make_fv3_faithful_pe_config`.
- cross-face vector halo rotation: LIVE `pad_halo_vector_4d` is ORTHOGONAL — DROPS
  FV3's `1/sin_sg(:,:,5)` non-orthogonality + z12/z21 cross-projection → real O(cosθ)
  seam error (largest AT seams). Faithful `pad_halo_dgrid_vector_4d` (12/12 tests) gated
  behind `use_fv3_cross_face_du_proj=False`.
**MAJOR gap:**
- `c_sw`+`d_sw` 2-stage SW core: production `CDGridShallowWaterModel` is single-stage
  SSP-RK3 over ONE Arakawa-Lamb centered tendency, NOT FV3 forward-backward time-split
  (c_sw dt/2→d_sw dt). Genuine 2-stage `fv3_fb_sw_step` exists but EXPERIMENTAL/unstable
  (~50 steps→NaN), unreachable from factory. ⇒ earlier "SW algorithmically faithful"
  was about the GATED FV3Edge path, not the production CDGrid SW.
- 3D PE D-grid vorticity/KE: **LIVE — the one non-orphaned major gap.** Centered
  `zeta_corner*v_d` + centered Bernoulli grad (primitive_eq_cdgrid:445), NOT FV3 upwind
  PPM `hord_vt`. Different algo class; legoESM compensates w/ explicit del-n damping vs
  FV3's implicit upwind enstrophy dissipation. Module docstring honestly says "Research
  path, not a faithful FV3 port"; factory tag "centered" is honest.
- ✅ **Ocean barotropic NOW HAS a faithful option** (19de9080):
  `barotropic_staggering="fv3edge"` routes through the TRUE FV3 edge-staggered SW
  core (FV3EdgeShallowWaterModel: upwind abs-vorticity flux + `_d2a2c_vect`).
  Gated; default stays validated `fv3sw` (centered). New cc↔edge lift round-trips
  0.1%. fv3edge validated: stable, rest ×4 machine-zero, geostrophic artifact
  resolved (nz 3.3%), 8/9. phillips is the only fail: the upwind scheme
  dissipates the baroclinic instability differently than latlon/fv3sw — REAL
  matrix phillips eta_growth 120→12.1, 130→11.4 (slow), so fv3edge needs a much
  higher (non-latlon-matching) div_damp to pass; NOT a clean 9/9 default
  replacement. So fv3sw (centered, validated 9/9, phillips 0.44≈latlon) stays
  DEFAULT; fv3edge is the available faithful upwind option. Test
  `test_fv3edge_barotropic.py`. (Earlier ~150-180 estimate used wrong IC.)
  CODEX-APPROVED: review confirmed cc↔edge lift indexing, rotation signs,
  _edge_to_cc inverse, fix_mass=False, no retrace all correct (only a doc-wording
  fix re per-substep vs sub-RK3-stage masking).

## ACTION QUEUE (from iter ~57 workflow synthesis)
SAFE-BOUNDED (no risk to equiangular default; do in loop iters):
- [DONE iter57] Fix stale failing test iter-707 (`test_cdgrid_fv3_regression.py`
  Iter707): was `places=4` abs (~5e-10 rel) on 1e5-mag KE → spurious fail on 6e-9
  float drift. Now rtol=1e-6, L2-scaled for the cancelling sum. NOT a 2nd→4th-order
  switch (agent misdiagnosed; value moved 6e-9 not O(%)). 2/2 pass.
- [TODO] Per-face + magnitude-normalized + cross-face seam-jump metric → replace
  pooled `compute_edge_artifact_metric` (halo.py:2994) + regression test (closes
  edge-metric lenses 1/2/4 permanently).
- [TODO] W2/W5 resolution+duration sweep (C36/C48/C96, W2→5d W5→15d) — verify the
  SW seam imprint grows or converges; W5 v is PHYSICAL flow (test the 5.95× claim).
- [TODO] Visual PNG inspection (CLAUDE.md-mandated; needs human/user — assistant
  barred from Read images): snapshots_v_native.png / snapshots_wind_speed_native.png
  for W2 + baroclinic.
- [TODO] Numerical regression locking the orphaned faithful ports (gnomonic_ed vs
  numpy FV3 ref; a2b_ord4 vs hand FV3 duogrid cascade). Fix latent `hord==9` mislabel.
FOUNDATIONAL (multi-day, re-calibration; NOT in a bounded loop iter):
- 3D PE upwind faithfulness ⇒ FV3 2-stage c_sw/d_sw rewrite (core dycore).
- Ocean→FV3Edge: faithful upwind option DONE (gated, 19de9080, validated 8/9).
  Promotion to default blocked by phillips: upwind baroclinic-instability
  dissipation differs from latlon → no div_damp cleanly gives 9/9 AND latlon-match.
  fv3sw stays validated default. (fv3edge usable now for upwind-faithful runs.)
- Climate-equilibrium cross-grid (held_suarez/AMIP): INFEASIBLE here, confirmed —
  cube HS ~1 min/day (30-day ~30min) + spectral ~15min, AND a day-30 run is
  non-equilibrated (HS climate needs ~200-day spin-up + time-averaging to compare
  statistically). latlon/ico hydro worse (~35min/case). DON'T re-attempt at this
  compute; cube climate-relevant dynamics (baroclinic eddies, gravity waves) are
  verified clean + close via the fast cases.
- C-face OMIP-coastline mask (coastal-BC, beyond matrix; SW-core jit-static).
State persisted: memory `cube-fv3-faithfulness-state`. Residuals: atm
baroclinic+topo = smooth converging truncation (cube more diffusive at C36);
SW cube W2 v_ll_Linf 0.34.
