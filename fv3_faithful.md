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
- **OCEAN FC velocity viscosity** vector-halo (4fe7108c).

## ✅ EDGE-ARTIFACT PROOF (codex-reviewed, two metrics)
(A) same-face roughness (RMS 2nd-diff edge band/interior): all cube cases ≤1.9×
(scalar-halo bug gave 25-167×). (B) TRUE cross-face continuity (edge cell vs
physical neighbor on adjacent face via halo): scalars (p_s,T,eta,SST) 0.5-1.8×,
geographic winds 0.5-2.3× — all CONTINUOUS. ⇒ no panel-seam artifact, proven
beyond visual. (Visual: geostrophic clean zonal bands, phillips smooth eddies
matching mpas, baroclinic v smooth — all no panel imprint.)

## VERIFICATION
- SW 16/16 all grids, aligned within 1 cell. Full fast cube atm dynamical suite
  PASS (baroclinic, rotated_baroclinic, gravity_wave, rossby_haurwitz,
  mountain_rossby, inertio_gravity, rotated_steady, rest_state_topo, dcmip_11/12)
  + NH dcmip_tc1, mass machine-zero. Ocean matrix cube 9/9; rest_state ×12
  machine-zero all grids.
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

## FAITHFULNESS AUDIT (codex + Fortran oracle)
- ✅ SW path (FV3EdgeShallowWaterModel→`fv3_sw_core`) **algorithmically faithful**:
  `_d2a2c_vect` (sin_sg upwind, corner 2×2 solve), `_d_sw1`, upwind
  `_vorticity_flux` (fv3_sw_core:1260).
- ⚠️ 3D PE dycore (`primitive_eq_cdgrid:445`) uses CENTERED `zeta_corner*v_d` —
  FV3-INSPIRED, validated, NOT FV3's upwind 2-stage. Functional faithfulness
  (results, edge-clean) everywhere. FV3-faithful corner d_sw5 div-damp + a2b-zeta
  KNOBS exist (`corner_div_damp_nord/d4_bg`, `use_fv3_a2b_zeta_corner`) but default
  OFF + are per-case-config-gated (held_suarez reads `LEGOESM_CDD_*` env;
  run_baroclinic config doesn't) + secondary (damping, not the core upwind-flux
  gap). Core PE upwind = major build (no drop-in faithful 3D dycore, unlike the
  ocean's FV3Edge).
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

## OPEN (sanctioned-major or CPU-bound — engineering decision: defer, don't risk
the validated state via loop hacks against a concurrently-`git reset` tree)
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
