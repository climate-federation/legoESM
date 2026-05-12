# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics code across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop with `/codex:adversarial-review`.

## Scope (parameterizations under review)

### Atmosphere — `src/legoesm/atmosphere/physics/`
- radiation: gray.py, rrtmgp_radiation.py, ozone_ml.py, solar.py, integration.py
- convection: bechtold.py, dca.py, emanuel.py, kain_fritsch.py, kuo.py, mass_flux.py, sbm.py, tiedtke.py, zhang_mcfarlane.py, integration.py
- microphysics: kessler.py, morrison.py, sundqvist.py, thompson.py, seifert_beheng.py, ml_emulator.py, integration.py
- turbulence: clubb_lite.py, edmf.py, holtslag_boville.py, louis.py, smagorinsky.py, surface_layer.py, tke.py, vertical_diffusion.py, ysu.py, pbl_height.py, integration.py
- clouds: cloud_fraction.py
- gravity_wave_drag, learned_column.py, thermodynamics.py, _shared.py

### Land — `src/legoesm/land/`
- multilayer_land.py, richards.py, slab_land.py, snow_budget.py, soil_grid.py, soil_hydraulics.py, soil_thermal.py, stomata_utils.py, surface_params.py, tridiag.py, carbon/

### Ocean — `src/legoesm/ocean/physics/`
- vertical_mixing/, bottom_drag/, lateral_mixing/, convection/, surface_forcing/, mixing.py, mpas_physics.py, shortwave_penetration.py, combined.py, mpas_physics.py

### Cryosphere — `src/legoesm/ice/`
- sea_ice.py, dynamics.py, itd.py, rheology.py, transport.py

## Iteration Log

(Iterations appended below; compressed every 10 iterations.)

### Iteration 1 — 2026-05-12

**Codex adversarial review (top-10 findings):**

1. `atmosphere/physics/microphysics/output.py:144` — explicit sedimentation can remove more condensate than exists when `V_t·dt/dz > 1`; produces negative hydrometeors. **Fix:** subcycle or limit outgoing flux by in-column mass + incoming flux.
2. `atmosphere/physics/convection/mass_flux.py:85` — re-derives dry hydrostatic dz/ρ instead of moist shared helpers. **Fix:** thread q_v into `_compute_column_geometry` and use `_shared` moist layer routines.
3. `atmosphere/physics/radiation/rrtmgp/rrtmgp.py:635` — H2O VMR divides by `1 - q_v` after lower-clip only; high q_v gives singular VMR. **Fix:** clip q_v < 1-eps before VMR conversion; centralize.
4. `land/multilayer_land.py:311` — spatial `land_params` makes `root_frac` shape `(ncol,n_layers)`, then `root_frac[None,:]` adds bogus leading axis. **Fix:** drop the unsqueeze and verify shapes.
5. `ocean/physics/convection/enhanced_diffusion.py:66` — K_conv=1.0 m²/s explicit centered diffusion, no dt/CFL limiter. **Fix:** route through implicit solver or cap/subcycle K from dt+dz.
6. `ocean/physics/surface_forcing/bulk_formulas.py:15` — `_P_ATM=101325.0` hardwired, ignores actual surface pressure. **Fix:** use `constants.p_ref`; ideally thread real `p_s`.
7. `ice/sea_ice.py:363` — dynamic multi-category thermo calls `_thermo_single` w/o configured shflx/lhflx; state uses simple_bulk while diagnostics may use MOST/COARE. **Fix:** compute one bulk-flux set; reuse.
8. `ice/sea_ice.py:625` — dynamic path returns zero `freshwater_flux`, `ocean_heat_extraction`, ocean stress. **Fix:** compute from pre/post Δh, basal heat, ice-ocean stress (mirror slab path).
9. `ice/itd.py:304` — ITD remap clamps mean thickness without adjusting area/volume; 1-4% volume drift per call. **Fix:** conservative Lipscomb/CICE remap or rescale area to preserve sum(a·h).
10. `ice/transport.py:110` — centered transport nonmonotone; post-step clamps destroy conservation. **Fix:** conservative monotone upwind/FCT with CFL enforcement.

**Actions this iteration (5 fixes applied):**
- [x] #6 `bulk_formulas.py:15` — `_P_ATM=101325.0` hardcode → added `constants.p_atm_std=101325.0` and use it (constants.py).
- [x] #3 `rrtmgp.py:631` — upper-clip `q_v` to 0.99 so `(1-q_v)` is bounded away from 0 in VMR conversion.
- [x] #4 `multilayer_land.py:311` — dropped bogus `root_frac[None,:]` unsqueeze (broke shape under 2D `lp`).
- [x] #1 `microphysics/output.py:112` — `sedimentation_tendency` now takes optional `dt` and applies positivity-preserving flux limiter `flux ≤ q·ρ·dz/dt`. All 4 callers (kessler/morrison/seifert_beheng/thompson) now pass `dt`.
- [x] #5 `enhanced_diffusion.py` + `convection/config.py` — explicit branch now caps K at `cfl_safety * dz_min² / cfl_dt_estimate` (defaults 0.45, 3600s). Implicit branch unaffected (unconditionally stable).

**Tests (post-fix):**
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py` → 58/58 pass.
- `tests/ocean/unit/test_mpas_physics.py` + surface_forcing + plume + physics_ocean → 47/47 pass.
- `tests/land/unit/test_multilayer_land.py` + land_ice_multilayer + diff_land → 74/74 pass.
- `tests/atmosphere/hydrostatic/unit/test_radiation*.py` → 71/71 pass.
- Atmosphere hydrostatic unit suite → 635 pass, 2 skip, 1 pre-existing fail (`test_hybrid_tracer_path_calls_vertical_advection` — hardcoded foreign path, not from us).
- Sea ice unit suite → 77 pass, 1 pre-existing fail (`Test8i_StefanBoltzmann::test_lw_up_matches` — confirmed via git-stash test on baseline), 1 xfail.

**Remaining from codex top-10 (deferred to next iterations):**
- #2 convection moist hydrostatic geometry (mass_flux.py:85)
- #7 sea_ice dynamic multi-cat thermo flux mismatch (sea_ice.py:363)
- #8 sea_ice dynamic ocean feedback zeros (sea_ice.py:625)
- #9 ITD remap non-conservative (itd.py:304)
- #10 sea-ice transport centered + clamps (transport.py:110)
