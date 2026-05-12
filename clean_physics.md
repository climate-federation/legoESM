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

### Iteration 2 — 2026-05-12

**Actions:**
- [x] #2 `mass_flux.py:70` — `_compute_column_geometry` now delegates to shared `compute_layer_dz`/`compute_rho` from `_shared.py` and accepts optional `q_v`; both callers (mass-flux closure + EDMF) now pass `q_v` so moist columns use virtual-T (~1 % thicker / less dense).
- [x] #7 `sea_ice.py` — new `_bulk_flux_dispatch(T_ice, forcing, config, U_min)` helper centralises MOST/COARE/Large-Yeager/simple_bulk dispatch.  Slab path and dynamic multi-category `_thermo_cat` both consume it, so state and diagnostic responses always use the configured scheme.
- [x] #8 `sea_ice.py:_build_response` — adds optional `h_old`, `ocean_sst`, `ocean_u`, `ocean_v`, `dt` kwargs; when supplied (dynamic path) computes `freshwater_flux`, `ocean_heat_extraction`, `ocean_stress_x/y` using the same formulae as the slab path, instead of zero placeholders.  Dynamic path threads pre-step aggregated `h_agg_initial` for the FW budget.
- [x] #9 `itd.py:linear_remap` — replaced the volume-leaking lo/hi clamp with a volume-conserving rescale: `a_post · h_post = a_pre · h_pre`.  Concentration capped at 1 (residual error only at the unphysical hi=100 m sentinel).  The previously-strict-xfail conservation test (`test_strict_volume_conservation_under_clamping`) now passes; xfail marker removed.

**Deferred:**
- #10 sea-ice transport centered + post-clamps: full FCT / conservative upwind needs a multi-file structural refactor; queued for iter-3.

### Iteration 3 — 2026-05-12

**Actions:**
- [x] #10 `ice/transport.py` — replaced the centered `divergence_3d` transport with conservative monotone PPM (Colella–Woodward with limiter) via `core.operators_fv.fv_flux_divergence` (2D path) and `core.operators_3d.fv_flux_divergence_3d` (multi-category path).  Volume / concentration / enthalpy each go through PPM-with-limiter so the scheme is monotone (no new extrema, no negative `h*a`, no `a > 1`); the post-step `jnp.clip` becomes a round-off safety guard rather than a conservation-breaker.  All 63 sea-ice transport / dynamics / differentiability tests pass.

**Tests (post-fix):**
- `tests/unit/test_land_ice_sea_ice_transport.py` → 2/2 pass.
- `tests/unit/test_sea_ice_dynamics.py` → 50/50 pass (including the previously-strict-xfail `test_strict_volume_conservation_under_clamping`).
- `tests/unit/test_diff_sea_ice.py` → 11/11 pass.

### Iteration 4 — 2026-05-12

**Codex iter-3 review surfaced 8 follow-up findings:**

1. `transport.py:113` — PPM still needs CFL subcycling; large `dt*u/dx` can overshoot and the final clip breaks conservation.
2. `sea_ice.py:647` — FW/heat response uses `(h - h_old)` which includes transport+ITD redistribution; should use thermodynamic ΔV only.
3. `sea_ice.py:583` — Multi-cat lead-freezing may double-deposit growth.
4. `itd.py:320` — Conservation breaks when rescaled `a > 1`; should redistribute residual, not clip.
5. `microphysics/output.py:150` — Precip diagnostic uses raw bottom flux, not the dt-limited one.
6. `_warm_rain.py:262` — Rain evaporation not donor-limited against `q_r`.
7. `enhanced_diffusion.py:68` — CFL cap uses `cfg.cfl_dt_estimate` instead of runtime `dt`.
8. `mass_flux.py:94` — `z` for full levels offset by ~½ layer.

**Actions (this iteration):**
- [x] #5 `microphysics/output.py:sedimentation_tendency` — new `return_surface_flux=True` kwarg returns the dt-limited bottom flux; all four schemes (kessler, morrison, thompson, seifert_beheng) now derive `precipitation` from this rather than `q_r·ρ·V_t` (which exceeds actual deposited mass when limiter fires).
- [x] #6 `_warm_rain.py:rain_evaporation` — new optional `dt` kwarg donor-limits evaporation against available `q_r` (`evap·dt ≤ q_r`). All callers (kessler — newly refactored to use the shared helper — plus thompson, morrison, seifert_beheng) pass `dt`.
- [x] #8 `mass_flux.py:_compute_column_geometry` — full-level height now uses `cumsum(dz)[::-1] - 0.5·dz` (mid-layer) instead of the layer-top.  Earlier value biased parcel-ascent diagnostics by ~½ layer (10–250 m).
- [x] #4 `itd.py:linear_remap` — when the volume-rescale would put `a > 1`, the residual is now folded back into thickness (`h = a_pre·h_pre`) so volume is preserved.  Earlier `jnp.clip(a, 0, 1)` discarded the excess.  Volume-conservation test continues to pass.

**Deferred to iter-5 (codex iter-3 findings #1, #2, #3, #7 + narrow review):**
- transport.py PPM CFL subcycling (`lax.scan` to CFL ≤ 1).
- sea_ice freshwater/heat from thermodynamic ΔV only (exclude transport/ITD redistribution).
- sea_ice multi-cat lead-freezing volume consistency.
- enhanced_diffusion: thread real `dt` instead of `cfl_dt_estimate`.
- ocean mixing.py / lateral_mixing CFL caps (5 findings).

**Tests (post iter-4):**
- microphysics + convection + sea-ice unit tests → 182/182 pass.
- atmosphere hydrostatic integration → 17/17 pass.

### Iteration 5 — 2026-05-12

**Actions:**
- [x] #7 enhanced_diffusion `dt` thread — `enhanced_diffusion_convection` now accepts an optional explicit `dt`; when provided it supersedes `cfg.cfl_dt_estimate` for the CFL cap.  Backward-compatible (default `None` keeps prior behaviour).
- [x] Ocean `mixing.vertical_diffusion` + `vertical_diffusion_variable_K` — both accept optional `dt` + `cfl_safety` (default 0.45) and cap the (scalar or per-interface) diffusivity by `cfl_safety · min(dz_k, dz_{k+1})² / dt`.  Default `dt=None` keeps existing callers unchanged.
- [x] #2 sea_ice FW/heat from thermodynamic ΔV — `_step_dynamic` now snapshots `h_agg_post_transport` (after horizontal advection, before thermo) and passes that to `_build_response.h_old` so the FW flux excludes horizontal transport (which conserves ice mass globally).

**Deferred to iter-6:**
- #1 sea-ice PPM CFL subcycling.
- #3 multi-cat lead freezing volume consistency.
- Ocean lateral_mixing harmonic/biharmonic CFL caps (need `grid.dx_min` plumbing).
- GM/Redi bolus Courant limiter.

**Tests (post iter-5):**
- sea-ice unit suite → 69/69 pass.
- ocean physics suite → 67/67 pass.

### Iteration 6 — 2026-05-12

**Actions:**
- [x] Sea-ice transport PPM subcycle — new `n_subcycles` kwarg on `advect_ice_tracers` runs `n_subcycles` PPM substeps with `lax.scan`; new `SeaIceConfig.transport_subcycles` (default 1) wires the choice through.  Default keeps current behaviour; setting > 1 enables CFL safety in storm conditions.
- [x] #3 Multi-cat lead freezing — new `_thermo_single` kwargs `open_water_fraction` and `enable_lead_freeze` allow the dynamic path to (a) drive lead-freeze concentration growth by the aggregated `(1 - sum_k conc_k)` instead of per-category `(1 - conc_k)` (which sums to > 1), and (b) fire the open-water freeze ONLY in category 0.  Implemented by splitting the `vmap` over the leading category vs the rest.

**Tests (post iter-6):**
- sea-ice unit suite → 78/79 pass (1 pre-existing Stefan-Boltzmann thermo failure unrelated to this work).
- atmosphere hydrostatic integration → 17/17 pass.

**Remaining (iter-7+):**
- Ocean lateral_mixing harmonic/biharmonic CFL caps.
- GM/Redi bolus Courant limiter.
- Visual / long-run validation of the new feedbacks.

**Tests (post-fix):**
- atmosphere microphysics + convection + land multilayer: 172/172 pass.
- sea ice unit suite: 78 pass, 1 pre-existing thermo failure (`Test8i_StefanBoltzmann::test_lw_up_matches` — confirmed pre-existing in iter-1).
- ocean + physics_convection suite: 1189 pass, 5 pre-existing failures (4× `test_ah_lat_scaling`, 1× `test_dca_extended_mse_conservation` — all confirmed pre-existing via baseline stash).
