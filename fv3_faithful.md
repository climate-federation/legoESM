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

### ✅ RESOLVED (commit bd74c45b) — geostrophic_adjustment cube non-zonal artifact
**FV3-faithful fix, never A-grid.** The cube barotropic defaulted to an **A-grid**
solver whose computational pressure mode grew the cube eta to **40% non-zonal**
(zonal `+5°C·cos lat` IC; user-confirmed visually) vs latlon 0%/mpas 0.3%. Routed
the barotropic free-surface mode (a 2-D SW system) through the validated FV3 cube
SW core `CDGridShallowWaterModel.step` (vector-invariant absolute-vorticity-FLUX
Coriolis w/ `cdgrid.f_corner`, SSP-RK3, div-damp + hyperdiff; W2/W5-clean).
New `barotropic_substeps_fv3sw` + `barotropic_staggering="fv3sw"` (OceanModel),
SW model built once in `__init__`. `h=H_bathy+eta`, `h_s=-H_bathy` ⇒ PGF=g·∇eta.
**Result: cube eta non-zonal 40%→1.5%** (latlon 0%, mpas 0.3%; visual = clean
zonal bands matching latlon/mpas). Cube ocean matrix **9/9 PASS** (was 8/9):
also FIXED barotropic_wave (a_grid over-damped 0.038<0.1 → now 0.857); geostrophic
max_speed 0.0242→0.0142 ≈ latlon 0.0159/mpas 0.0169; rest_state ×4 stay
machine-zero. Ocean-tuned `barotropic_sw_div_damp_factor=120` (atm preset 8 too
weak ⇒ phillips over-grew 24× → now eta_growth 8.8, max_eta 0.44 ≈ latlon 0.41).
Test `tests/ocean/unit/test_fv3sw_barotropic.py`.
**Solution-space (why this is THE fix):** a_grid = 40% artifact; explicit-f*v
c_grid = NaN (`fv3_cc2c` breaks FB skew-symmetry); bare vector-invariant under
forward-Euler OR RK3 = NaN; only the full SW core (RK3+div_damp+hyperdiff) is
stable AND zonal (isolated zonal-eta test nz=0.0000).
**Codex review (confirmed correct for wet cells: height mapping, no stale-JIT,
rest_state exactly preserved, baroclinic reconstruction OK). 2 LAND follow-ups
for real-coastline OMIP (not the idealized 9/9 cases):** (1) coasts not
impermeable DURING substeps — the SW core is global/land-free, mask applied only
AFTER the substep loop ⇒ water/momentum can cross coasts mid-loop (rest exact;
non-rest coastal basins not perfectly closed). (2) the SW per-substep mass fixer
conserves land-INCLUSIVE `sum(h·area)`, not wet-ocean `sum(eta·mask·area)`.
NEXT: mask SW winds at coast corners inside the fori_loop + use the ocean's
masked conservation domain for the fast mode. Polar-cap land at C24/1-day leaks
below test thresholds (geostrophic PASS, nz 1.5%), so idealized cases are safe.

## SW visual verdicts
1. cosine-bell day-1 = PPM-limiter interior diffusion (cube/ico L2 1.4×), not edge.
2. W5 propagates ≈ latlon/MPAS by day 15; ~12% day-1 damping deficit.
3. alignment FIXED (above).

## Residuals (distinct from the fixed panel-edge artifact)
- atm baroclinic+topography = SMOOTH CONVERGING gradient-truncation (only the
  FC-spectral gradient build shrinks it at coarse C36; large/uncertain).
- SW cube W2 v_ll_Linf 0.34 = separate FV3Edge path.
