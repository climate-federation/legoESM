# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iterations 1-29 — Summary (compressed 2026-05-12 after iter-29)

### Constants & shared utilities
- `constants.p_atm_std = 101325 Pa` and `constants.R_universal = 8.314462618`
  (CODATA 2018) added.  `ocean/surface_forcing/bulk_formulas.py` and
  `land/carbon/stomata.py` now reference them instead of local literals.
- `convection/mass_flux._compute_column_geometry` delegates to shared
  `_shared.compute_layer_dz` / `compute_rho` with `q_v` for virtual-T moist
  geometry.  Threaded through all 5 mass-flux schemes (Bechtold, Emanuel,
  Tiedtke, Kain-Fritsch, Zhang-McFarlane).  Full-level heights use mid-layer
  `cumsum(dz)[::-1] - 0.5·dz`.
- Kessler microphysics uses shared `_warm_rain.rain_evaporation` helper.
- Constant-hygiene audit (iter-12): all `g`/`R_d`/`c_pd`/`L_v`/`L_s`/`L_f`/
  `sigma_sb`/`T_freeze`/`Omega`/`R_earth`/`rho_air`/`rho_ocean`/`epsilon`
  references go through `constants.py`.

### Conservation & monotonicity
- **Sedimentation** (`microphysics/output.py`): optional `dt` applies
  Bott/FCT positivity flux limiter (`flux ≤ q·ρ·dz/dt`).  New
  `return_surface_flux=True` returns the dt-limited bottom flux for
  conservative precipitation diagnostics.  New `extra_sink` parameter
  jointly caps sed + rain-evap against `q_r/dt`.  Threaded through
  kessler/morrison/thompson/seifert_beheng.
- **Rain evaporation** (`_warm_rain.py`): optional `dt` clamps evaporation
  to `q_r/dt`.  All four schemes pass `dt`.
- **Sundqvist autoconversion**: donor cap so `P_auto · dt ≤ q_c`.
- **Sea-ice transport**: replaced centered divergence with conservative
  monotone PPM (Colella–Woodward with limiter); `n_subcycles` runs PPM
  substeps with `lax.scan`.  `SeaIceConfig.transport_subcycles` exposes.
- **ITD remap**: volume-conserving rescale `a_post · h_post = a_pre · h_pre`;
  residual folded into thickness when `a > 1`.
- **RRTMGP**: upper-clip `q_v ≤ 0.99` so the `(1 - q_v)` denominator in the
  H2O VMR conversion is bounded.
- **Morrison + Thompson vapor donor clamp**: jointly clamp `cond_pos +
  dq_i_dep` against `q_v/dt` (codex iter-25 + iter-29).
- **DCA STANDARD MSE conservation** (test updated to match in-scheme latent
  heating in `delta_T_lh`).

### CFL caps & stability
- **Ocean enhanced_diffusion convection**: explicit branch delegates per-
  interface CFL cap to `vertical_diffusion_variable_K(dt=...)` using
  `dz_actual = dz_ref · jacobian`.  Optional explicit `dt` precedes
  `cfg.cfl_dt_estimate`.
- **Ocean vertical mixing leaf** (`mixing.py`): `vertical_diffusion` and
  `vertical_diffusion_variable_K` accept optional `dt` + `cfl_safety`.
- **Ocean vertical-mixing schemes** (KPP, Richardson, constant) thread
  `dt` to the leaf so the CFL cap is reachable from every entry point.
- **Ocean lateral mixing harmonic + biharmonic**: opt-in `enforce_cfl` flag
  with `cfl_dt_estimate` / `cfl_safety` config fields; harmonic
  `A_h`/`K_h ≤ cfl_safety · dx² / dt` (default safety 0.20), biharmonic
  `B_h_* ≤ cfl_safety · dx⁴ / dt` (default safety 0.05).
- **EDMF mass flux** (`turbulence/edmf.py`): caps `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- **Sea-ice `_bulk_flux_dispatch`** helper centralises MOST/COARE/
  Large-Yeager/simple_bulk dispatch.  Used by slab path, dynamic
  multi-category `_thermo_cat`, and diagnostic `_build_response`.
- **`_build_response` ice → ocean feedback**: optional `h_old`, `ocean_*`,
  `dt` kwargs compute `freshwater_flux`, `ocean_heat_extraction`,
  `ocean_stress_x/y` (per-ice-area thickness rate; full V=h·A CICE
  bookkeeping deferred).  Dynamic path threads `h_agg_post_transport`
  (post-transport, pre-thermo) so FW = thermodynamic ΔV only, no
  transport redistribution.
- **Multi-cat lead freeze**: `_thermo_single` gains `open_water_fraction`
  + `enable_lead_freeze` kwargs.  Dynamic dispatch drives concentration
  growth by aggregated `(1 - sum_k conc_k)` and fires open-water freeze
  in category 0 only.
- **Sea-ice radiative emission**: `lw_up = ε·σ·T⁴ + (1-ε)·lw_down`
  (grey surface, reflected-LW component included; test updated).
- **Multilayer land** dropped bogus `root_frac[None,:]` unsqueeze.
- **Multilayer land dew/deposition routing** (iter-23+24): when
  `has_snow=False`, dew is routed to `flux_top` (no transpiration sink);
  when `has_snow=True`, snowpack absorbs the entire latent flux and soil
  sees zero direct flux.

### Tracer-mixing land mask (iter-25, iter-27)
- `ocean/physics/lateral_mixing/harmonic.py` + `biharmonic.py` output T/S
  tendency is multiplied by `mask_3d` so land cells receive zero tendency.
  Full no-flux BC (replicate ocean values at land neighbours before the
  operator) requires an operator-stencil refactor; output mask is the
  safe minimum.

### Test fixes
- `Test8i_StefanBoltzmann::test_lw_up_matches` — grey-surface LW.
- `test_hybrid_tracer_path_calls_vertical_advection` — relative path.
- `test_ah_lat_scaling` (4 tests) — pass `power=2` for legacy cos² coverage.
- `test_dca_extended_mse_conservation` — STANDARD MSE (in-scheme latent heat).
- 7 / 7 pre-existing failures resolved (1 in iter-13, 6 in iter-14).

### Regression tests (iter-8 + iter-22)
- `tests/ocean/unit/test_ocean.py::TestVerticalMixing::test_vertical_diffusion_cfl_cap_keeps_step_stable`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::test_sedimentation_cfl_positivity`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::test_sedimentation_surface_flux_conservation`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::TestSundqvist::test_autoconversion_does_not_drive_qc_negative`

### Test status (post iter-29)
- 17 / 17 atmosphere hydrostatic integration tests pass.
- 169 / 169 atmosphere convection + microphysics + ocean vertical mixing
  targeted tests pass.
- 95 / 95 atmosphere turbulence tests pass.
- 79 / 79 sea-ice unit tests pass.
- 75 / 75 land carbon + multilayer + diff_land tests pass.

## Inspected & clean (no fix needed)
- `land/snow_budget.py`, `land/carbon/carbon_cycle.py`, `land/richards.py`,
  `land/soil_thermal.py` — implicit / well-disciplined.
- `atmosphere/physics/turbulence/vertical_diffusion.py` — implicit.
- `atmosphere/physics/clouds/cloud_fraction.py` — shared
  `saturation_mixing_ratio`.
- `atmosphere/physics/gravity_wave_drag/lindzen.py` — `constants.*`,
  differentiable scan.
- `ocean/physics/shortwave_penetration.py` — conservation correction.
- `ocean/physics/bottom_drag/quadratic.py` — proper sign convention.
- `ice/rheology.evp_stress_update`, `ice/dynamics.evp_solver` — implicit
  / CICE convention.
- `atmosphere/physics/convection/{dca,kuo,zhang_mcfarlane}.py` — moist
  static-energy budgets verified.
- `atmosphere/physics/microphysics/seifert_beheng.py` — joint donor clamps.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu}.py`
  — all use shared `virtual_temperature`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter (codex narrow review).
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping (codex
  iter-22 #1 + iter-25 revert): h and conc evolve semi-independently
  in the current slab path, so `d(h·conc)/dt` mis-counts new lead ice
  (h vs h_new_ice) and double-counts melt-retreat.  Per-ice-area `dh/dt`
  is the current bookkeeping.
- No-flux BC for harmonic/biharmonic tracer Laplacian (replicate ocean
  values at land neighbours before the operator).  Output mask is the
  current containment.
- Threading `dt` into `physics_fn` across the 3 ocean PE backends + 3
  model drivers — production uses the implicit ocean solver
  (unconditionally stable), so the marginal stability gain is low.
- Visual / long-run validation of the new ice → ocean feedbacks.

## Next iterations
Continue addressing further codex adversarial-review findings until
physics implementation is provably conservative, monotone, and CFL-safe
across all parameterizations.

### Iteration 30 — 2026-05-12

Compression iteration.  Folded iter-20 through iter-29 into the
"Iterations 1-29 — Summary" section.  Detailed per-iteration narratives
remain in the commit messages on `clean_physics`.
