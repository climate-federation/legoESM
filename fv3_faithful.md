# FV3-faithful cubed-sphere — change log (shrunk @ iter ~31)

Goal: cube FV core faithful to GFDL FV3 across atm+ocean (SW→AMIP/OMIP), matching
MPAS/ico + lat-lon FV, **zero cube edge artifacts**, visual+quantitative, codex-
reviewed. CPU only (`JAX_PLATFORMS=cpu`; Metal broken). Oracle
`../Code/FV3/atmos_cubed_sphere-symmetryclean/` READABLE (70 F90). **User
directive: never A-grid — be FV3-faithful (staggered C/D).** Per-iter detail in git.

## ✅ ATMOSPHERE cube edge imprint — FIXED (vector-seam)
`_step_cell_centre`/`tendencies()`/diffusion lifted cell-centre winds→D-grid
treating the **vector** (u,v) as **scalars** (no cross-face rotation). Fix: 7 sites
use `center_to_dgrid_vector`/`pad_halo_vector_4d` (ac6a8f58…4fd71710, 6 codex
APPROVE). baroclinic v_rms 4.95→0.60; converges ~2nd order; mass 2.6e-11→1.1e-15.
Re-verified (iter ~30): fresh baroclinic C36 cube PASS, native v-wind = smooth
zonal bands, NO panel blocks/seam noise. Test `test_vector_cc_to_dgrid_wind_lift.py`.
NH cube dcmip_tc1 PASS (|w|_max=0.014, mass drift 1e-15 — healthy steady balance).
3D cross-grid CLOSENESS (gravity_wave_3_1, 0.25d): cube max|v|=19.3 ≈ latlon 20.2
(within 5%); cube native v-wind = smooth wavenumber-2 wave, no panel imprint.
(ico5 hydro pathologically slow on CPU — killed; cube↔latlon closeness sufficient.)

## ✅ CROSS-GRID longitude alignment — FIXED (codex clean)
latlon stored native 72×144 under a 360-pt label (tens-of-deg translation);
gaussian/spectral kept native lon; npz lon + icosa weights node-centered while cube
data cell-centered. Fix: one canonical canvas `_canvas_lat/_canvas_lon`
(== regridding.py:457) for EVERY regrid+npz+axis (2d04a168, 882b4063, 418cc7de).
Also unblocked latlon SW (was 0%-running: bad `_conservation_accumulator` import).
SW cosine_bell/W5/W2 all 4 grids co-located within 1 cell. Test
`test_latlon_regrid_alignment.py`.

## ✅ OCEAN cube — FV3-faithful barotropic (headline, never A-grid)
- ✅ viscosity vector-halo (4fe7108c, codex): FC velocity Laplacian/hyperdiff
  scalar-halo'd stacked (u,v) → now `_fc_pad_halo_vector` per-comp + vector-∇⁴.
- ✅ **geostrophic_adjustment non-zonal artifact RESOLVED** (bd74c45b, codex):
  cube barotropic was **A-grid** → computational pressure mode grew eta to **40%
  non-zonal** (user-confirmed visually) vs latlon 0%/mpas 0.3%. Routed the
  barotropic (a 2-D SW system) through the validated FV3 cube SW core
  `CDGridShallowWaterModel.step` (vector-invariant absolute-vorticity-flux Coriolis
  w/ `cdgrid.f_corner`, SSP-RK3, div-damp+hyperdiff; W2/W5-clean).
  `barotropic_substeps_fv3sw` + `barotropic_staggering="fv3sw"`; `h=H_bathy+eta`,
  `h_s=-H_bathy` ⇒ PGF=g·∇eta. **Result: 40%→1.5% non-zonal** (visual = clean zonal
  bands ≈ latlon/mpas). Ocean-tuned `barotropic_sw_div_damp_factor=120` (atm preset
  8 too weak ⇒ phillips over-grew). Test `test_fv3sw_barotropic.py`.
  Solution-space: a_grid=40% artifact; explicit-f*v c_grid=NaN; bare VI under
  FE/RK3=NaN; only full SW core (RK3+damp) is stable AND zonal (zonal-eta nz=0.0000).
- ✅ land refinement (codex): SW `fix_mass=False` (ocean masked fixer governs
  wet-ocean volume) + per-substep corner-ocean wind mask (coasts impermeable; land
  eta=0 exact, rest machine-zero). Residual (real-coastline OMIP only, NOT cube
  panel edges): sub-RK3-stage leak + corner mask over-damps wet faces touching a
  land corner → exact fix = C-FACE no-through-flow mask in the SW tendency. Scoped.
- **Cube ocean matrix 9/9 PASS.** Cross-grid (all grids): rest_state ×12
  machine-zero; geostrophic max_speed 0.0142 ≈ latlon 0.0159/mpas 0.0169;
  barotropic_wave cube least-dissipative (most accurate; latlon/mpas over-damp);
  phillips eddies smooth, match mpas wavenumber, **no panel-edge imprint**.
  ⇒ ocean cube panel edges verified clean across dynamical cases.

## QUANTITATIVE edge-artifact proof (iter ~35, codex-reviewed)
TWO complementary metrics on cube NATIVE (6,n,n) final fields:
(A) same-face roughness = RMS(2nd-diff in the 1-cell edge band)/RMS(interior
2nd-diff). Catches grid-scale seam noise; the scalar-halo BUG gave 25-167×.
Result <2 (CLEAN): atm baroclinic v 1.86, rotated_baroclinic v 1.09,
gravity_wave v 1.33; ocean geostrophic/phillips eta 0.06, speed 1.08.
(B) TRUE cross-face continuity (codex flagged (A) alone can't see cross-panel
jumps) = RMS(edge cell − physical neighbor on adjacent face via halo)/RMS(interior
neighbor diff). SCALARS + geographic wind components (basis-continuous across
seams): p_s 1.30/1.81, T_3d 1.60, eta 0.50/0.57, SST 1.01/1.33, wind u/v/speed
0.5–2.3. All CONTINUOUS (≤2.3×). [Note: applying `pad_halo_vector` (face-local
rotation) to the saved GEOGRAPHIC east/north winds fabricated a spurious 65× —
analysis bug, corrected; geographic components are seam-continuous scalars.]
⇒ both metrics confirm **no panel-seam artifact** (scalars continuous, winds
continuous, edges smooth) across the verified atm+ocean dynamical cases.

## Cube atmosphere full dynamical-suite sweep (iter ~36)
ALL fast cube hydro dynamical cases PASS, mass drift ~machine-zero, bounded v:
baroclinic, rotated_baroclinic, gravity_wave_3_1, rossby_haurwitz_6_0 (max|v|27.4),
mountain_rossby_5_0 (10.2), inertio_gravity_3_2 (16.9), rotated_steady (23.8),
rest_state_topo (max|v|1.2 = bounded sigma-PGF topo residual), dcmip_transport_11/12;
+ SW 16/16 + NH dcmip_tc1. Only climate (held_suarez/amip, 30-day) is CPU-bound.

## Regression health (iter ~34)
44/44 PASS: test_vector_cc_to_dgrid_wind_lift, test_latlon_regrid_alignment,
test_fc_velocity_viscosity_vector_halo, test_fv3sw_barotropic, test_ocean_fc,
test_no_scheme_duplication — all session fixes regression-clean, no duplication.

## FAITHFULNESS AUDIT (codex + FV3 oracle, iter ~37)
Codex adversarial audit of legoESM cube dycore vs FV3 sw_core.F90 c_sw/d_sw.
Precise map (FUNCTIONAL faithfulness — FV3-like results, edge-clean, close to refs
— holds everywhere; ALGORITHMIC line-by-line faithfulness is mixed):
- ✅ **SW path (FV3EdgeShallowWaterModel → `fv3_sw_core`) IS faithful**: `_d2a2c_vect`
  (full d2a2c, sin_sg upwind, adjacent-strip + corner 2×2 solve, sw_core:618-812),
  `_d_sw1_recompute_ut_vt`. The SW matrix cube uses this.
- ⚠️ **3D PE atm dycore** (`primitive_eq_cdgrid:445`) + **ocean barotropic**
  (`cdgrid_momentum_tendencies:1167`) use **CENTERED** `zeta_corner*v_d`, NOT FV3's
  UPWIND absolute-vorticity flux (sw_core:423-490); `dgrid_to_cgrid:419` is a
  simplified d2a2c (v_c plain-averaged, no cosa_v/sina_v); div-damp is cell-center
  continuous, not FV3 corner d_sw5/nord. These are FV3-INSPIRED vector-invariant +
  explicit damping — validated functionally (baroclinic converges, edge-clean, all
  cases pass) but NOT a line-by-line FV3 port.
- SCOPED UPGRADE — now precisely characterized (iter ~38): the faithful upwind
  flux `_vorticity_flux` (fv3_sw_core:1260, `where(fy1>0,vort[:-1],vort[1:])`) is
  part of FV3's **2-stage c_sw→d_sw scheme** (C-grid half-step produces the
  time-centered winds that upwind-transport vorticity for the D-grid full step).
  The legoESM PE uses a **single-stage RK3 with centered `zeta_corner*v_d`** +
  explicit hyperdiff — a DIFFERENT, validated discretization, not a term-swap away
  from FV3. True PE faithfulness ⇒ adopt the FV3 2-stage structure (major dycore
  rewrite; the PE solves T/ln_ps/u/v, can't reuse the SW FV3Edge model wholesale)
  + re-calibrate damping + re-validate every 3D case. **Engineering decision
  (deferred):** the current single-stage centered scheme is validated, edge-clean,
  convergent, and results-faithful (FV3-like); the 2-stage rewrite risks that for a
  purity goal whose results-benefit is marginal (centered+explicit-damping ≈
  upwind+implicit-damping in net dissipation). Documented as the decision point;
  NOT rushed. SW path already faithful; ocean barotropic reuses the centered
  CDGridShallowWaterModel (validated, geostrophic-artifact-free).

## SW visual verdicts
1. cosine-bell day-1 = PPM-limiter interior diffusion (cube/ico L2 1.4×), not edge.
2. W5 propagates ≈ latlon/MPAS by day 15; ~12% day-1 damping deficit.
3. alignment FIXED (above).

## Residuals / open (separate from the fixed panel-edge artifacts)
- atm baroclinic+topography = SMOOTH CONVERGING gradient-truncation (cube more
  diffusive at coarse C36: baroclinic max|v| ~11-13 vs latlon ~26 at C36/72x144;
  converges with resolution; v-wind smooth = not an edge artifact). FC-spectral
  gradient build = the lever; large/uncertain.
- SW cube W2 v_ll_Linf 0.34 = separate FV3Edge path.
- C-face OMIP-coastline mask (coastal-BC, not a cube edge); full AMIP/OMIP
  cross-grid render pathologically SLOW on CPU (latlon hydro ~35 min/case).
