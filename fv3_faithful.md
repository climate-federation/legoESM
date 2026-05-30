# FV3-faithful cubed-sphere — change log (shrunk @ iter ~22)

Goal: cube FV core faithful to GFDL FV3 across atm+ocean (SW→AMIP/OMIP), matching
MPAS/ico + lat-lon FV, **zero cube edge artifacts**, visual+quantitative, codex-
reviewed. CPU only (`JAX_PLATFORMS=cpu`; Metal broken). Per-iter detail in git.
**Oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/` NOW READABLE** (70 F90;
TCC unblocked iter ~22). **User directive: never A-grid — be FV3-faithful
(staggered C/D).**

## ✅ ATMOSPHERE cube edge imprint — ROOT-CAUSED + FIXED
`_step_cell_centre`/`tendencies()`/diffusion converted cell-centre winds→D-grid
treating the **vector** (u,v) as **scalars** (no cross-face rotation). Fix: 7
sites use `center_to_dgrid_vector` / `pad_halo_vector_4d` (commits ac6a8f58…
4fd71710, 6 codex APPROVE). baroclinic v_rms 4.95→0.60; converges (2nd order);
mass drift 2.6e-11→1.1e-15; v-field wavenumber-4 blocks→smooth bands. Regression
`tests/test_vector_cc_to_dgrid_wind_lift.py`. SW 16/16 all grids; HS 286K physical.

## ✅ CROSS-GRID longitude alignment — FIXED (codex clean)
latlon stored native 72×144 under a 360-pt label (tens-of-deg translation);
gaussian/spectral kept native lon; npz lon + icosa weights node-centered while
cube data cell-centered. Fix: one canonical canvas `_canvas_lat/_canvas_lon`
(== regridding.py:457) for EVERY regrid+npz+axis (2d04a168, 882b4063, 418cc7de).
Also unblocked latlon SW (was 0%-running: bad `_conservation_accumulator` import).
SW cosine_bell/W5/W2 all 4 grids co-located within 1 cell (visual). Test
`tests/test_latlon_regrid_alignment.py`.

## OCEAN cube
- rest_state ×4 (cube/latlon/mpas) PASS — cube eta drift 1e-23..1e-31 (FC-Gram
  machine-zero). phillips, inertia_gravity_wave, overflow cube PASS.
  barotropic_wave cube FAIL (min_amp 0.038<0.1) = PRE-EXISTING FC over-damping.
- ✅ Fixed (4fe7108c, codex clean): FC cube-ocean velocity viscosity halo'd
  stacked (u,v) with SCALAR `pad_halo_4d` → now `_fc_pad_halo_vector` per-comp +
  hand-built vector-∇⁴. Zonal-jet |∇²V| nz 0.59→0.38. Test
  `tests/ocean/unit/test_fc_velocity_viscosity_vector_halo.py`.

### ⚠️ OPEN — geostrophic_adjustment cube non-zonal artifact (user-confirmed)
Zonal IC (`+5°C·cos lat`) but cube eta **40% non-zonal** vs latlon 0%/mpas 0.3%.
ROOT CAUSE (measured): cube barotropic = **A-grid** (`staggering=a_grid`); latlon
c_grid, mpas TRiSK = immune. A-grid computational mode: eta→5% after step1, then
a wavenumber-1 mode grows 4%→40% over ~12h (e-fold ~1.7h ≈ 1/f). Seed = FC PGF
~2% imprint. NOT viscosity, NOT barotropic continuity div (already vector).
**Exhausted levers**: higher A-grid diffusion makes it WORSE (over-damps zonal
signal, nz 0.40→0.74); the C-grid solver (`barotropic_cgrid.py`) is UNSTABLE
(eta 0.05→17m→NaN by 11.5h) — **root-caused to its explicit-f*v Coriolis**
(f=0 test: 17m→0.84m, no blow-up; `fv3_cc2c` vector-rotation mixes u,v → destroys
the forward-backward skew-symmetry → effectively explicit → amplifies at ~f).

### FV3-FAITHFUL FIX (oracle-confirmed) — NEXT, refined by iter ~23 attempt
FV3 `c_sw` (sw_core.F90:405-490) Coriolis = **absolute-vorticity FLUX**, NOT
explicit f*v: `vort = fC + rarea_c·curl(uc·dxc, vc·dyc)`; transport upwind by
contravariant transverse flux; momentum `uc += fy1·fy − rdxc·Δ(KE)`. legoESM has
this for the atm: `cdgrid_momentum_tendencies` (vector-invariant w/ `cdgrid.
f_corner`) + `cgrid_mass_flux_divergence` (PPM continuity).
**ATTEMPTED iter ~23 (reverted — failed validation):** new D-grid forward-
backward barotropic (`barotropic_substeps_dgrid`) reusing those ops. rest_state
regressed 1e-31→2e-5; geostrophic **eta→27m→NaN by 11h even with div_damp=0.25**.
**ROOT LESSON (key):** a single-stage forward-backward only time-centers the
**gravity wave** (eta↔u); the **Coriolis** inside the vector-invariant tendency
is still forward-Euler → amplifies at ~(f·dt)² (same explicit-Coriolis blow-up as
the explicit-f*v c_grid). FV3's stability comes from its **c_sw/d_sw TWO-STAGE
time scheme** (the C-grid half-step time-centers the Coriolis), NOT single-stage
Euler. ⇒ **Correct fix**: drive the barotropic substep through the validated FV3
2-stage SW stepper `CDGridShallowWaterModel.step` (or replicate its c_sw→d_sw
half/full split): build a `CDGridShallowWaterState` from (h=eta+H_bathy or eta,
D-grid barotropic winds), step, extract eta + winds. Must reuse FV3's time scheme,
not hand-rolled Euler. Validate: rest_state machine-zero, geostrophic nz→~mpas,
barotropic_wave amplitude, stability, no regressions; codex review. (Never a_grid;
never explicit f*v; never single-stage Euler.)

## SW visual verdicts
1. cosine-bell day-1 = PPM-limiter interior diffusion (cube/ico L2 1.4×), not edge.
2. W5 propagates ≈ latlon/MPAS by day 15; ~12% day-1 damping deficit.
3. alignment FIXED (above).

## Residuals (distinct from the fixed panel-edge artifact)
- atm baroclinic+topography = SMOOTH CONVERGING gradient-truncation (only the
  FC-spectral gradient build shrinks it at coarse C36; large/uncertain).
- SW cube W2 v_ll_Linf 0.34 = separate FV3Edge path.
