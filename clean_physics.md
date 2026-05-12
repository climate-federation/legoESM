# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iterations 1-19 — Summary (compressed 2026-05-12 after iter-19)

### Constants & shared utilities (iter-1, iter-2, iter-12, iter-16)
- Added `constants.p_atm_std = 101325 Pa` (standard atmosphere) and switched
  `ocean/surface_forcing/bulk_formulas.py` from hardwired `_P_ATM`.
- `convection/mass_flux._compute_column_geometry` now delegates to shared
  `_shared.compute_layer_dz` and `compute_rho` with `q_v` for virtual-T moist
  geometry (~1% thicker / less dense in tropics). Full-level heights use
  the mid-layer `cumsum(dz)[::-1] - 0.5·dz` (iter-4).
- All five mass-flux convection schemes (Bechtold, Emanuel, Tiedtke,
  Kain-Fritsch, Zhang-McFarlane) now thread `q_v` through
  `_compute_column_geometry` (iter-16) — previously silently used dry
  geometry.
- Kessler microphysics refactored to use shared `_warm_rain.rain_evaporation`
  helper (iter-4).
- Constant-hygiene audit (iter-12): **clean** — no production-code violations
  of the constants.py discipline remain across `atmosphere/physics/`,
  `land/`, `ocean/physics/`, `ice/`.

### Conservation & monotonicity
- **Sedimentation (microphysics/output.py, iter-1/4)**: optional `dt` adds a
  Bott/FCT positivity flux limiter (`flux ≤ q·ρ·dz/dt`); new
  `return_surface_flux=True` returns the dt-limited bottom flux for
  conservative precipitation diagnostics. Threaded through kessler,
  morrison, thompson, seifert_beheng.
- **Rain evaporation (_warm_rain.py, iter-4)**: optional `dt` clamps the
  evaporation rate to `q_r/dt` (donor positivity). All four schemes pass `dt`.
- **Sea-ice transport (ice/transport.py, iter-3/6)**: replaced centered
  divergence with conservative monotone PPM (Colella–Woodward with limiter)
  via `core.operators_fv.fv_flux_divergence`; new `n_subcycles` runs PPM
  substeps with `lax.scan`. Wired through `SeaIceConfig.transport_subcycles`.
- **ITD remap (ice/itd.py, iter-2/4)**: replaced volume-leaking lo/hi clamp
  with a volume-conserving rescale `a_post · h_post = a_pre · h_pre`; when
  the rescale would put `a > 1` the residual folds back into thickness so
  volume is preserved exactly. Previously-strict-xfail conservation test now
  passes.
- **RRTMGP (radiation/rrtmgp/rrtmgp.py, iter-1)**: upper-clip `q_v ≤ 0.99` so
  the `(1 - q_v)` denominator in the H2O VMR conversion is bounded.

### CFL caps & stability (iters 1, 5, 7, 10, 11, 17)
- **Ocean enhanced_diffusion convection**: explicit branch caps `K` at
  `cfl_safety · dz_min² / dt`. Optional explicit `dt` argument takes
  precedence over `cfg.cfl_dt_estimate` when supplied.
- **Ocean vertical mixing leaf** (`mixing.py`): `vertical_diffusion` and
  `vertical_diffusion_variable_K` accept optional `dt` + `cfl_safety` and cap
  the diffusivity by `cfl_safety · min(dz_k, dz_{k+1})² / dt`.
- **Ocean vertical-mixing schemes** (KPP, Richardson, constant) now thread
  `dt` into the leaf calls so the CFL cap is reachable from every entry
  point. Default `None` preserves existing behaviour.
- **Ocean lateral mixing harmonic + biharmonic**: opt-in `enforce_cfl` flag
  with `cfl_dt_estimate` / `cfl_safety` config fields caps `A_h`/`K_h` by
  `cfl_safety · dx² / dt` (harmonic, default safety 0.20) and `B_h_*` by
  `cfl_safety · dx⁴ / dt` (biharmonic, default safety 0.05).
- **EDMF mass flux** (`turbulence/edmf.py`): caps `M = a·ρ·w_u` at
  `0.5·ρ·dz/dt` so the explicit centered-FD MF tendency cannot overshoot
  θ/q in deep convection.

### Coupler / feedback paths (iter-2/5/6)
- **Sea-ice** (`ice/sea_ice.py`): new `_bulk_flux_dispatch` helper centralises
  MOST / COARE / Large-Yeager / simple_bulk dispatch — used by slab path,
  dynamic multi-category `_thermo_cat`, and diagnostic `_build_response`.
- **`_build_response` ice → ocean feedback**: new optional `h_old`,
  `ocean_sst`, `ocean_u/v`, `dt` kwargs compute `freshwater_flux`,
  `ocean_heat_extraction`, `ocean_stress_x/y` using the slab-path formulae
  instead of zero placeholders. Dynamic path threads
  `h_agg_post_transport` (after horizontal advection, before thermo) so the
  FW flux is the thermodynamic ΔV only, not the transport redistribution.
- **Multi-cat lead freeze**: `_thermo_single` gains `open_water_fraction` and
  `enable_lead_freeze` kwargs. Dynamic dispatch drives concentration growth
  by aggregated `(1 - sum_k conc_k)` and fires the open-water freeze in
  category 0 only — no per-category double deposit.
- **Multilayer land** (iter-1): dropped bogus `root_frac[None,:]` unsqueeze
  that broke shape under 2D `land_params`.
- **Sea-ice radiative emission** (iter-13): test_lw_up_matches updated to
  account for the grey-surface reflected-LW component
  `lw_up = ε·σ·T⁴ + (1-ε)·lw_down` (test was missing the second term).

### Test fixes (iter-13, iter-14)
- `Test8i_StefanBoltzmann::test_lw_up_matches` — added missing reflected-LW term.
- `test_hybrid_tracer_path_calls_vertical_advection` — replaced hardcoded
  `/home/gentine/...` path with `Path(legoesm.__file__)`-based lookup.
- `test_ah_lat_scaling` (4 tests) — explicit `power=2` to preserve the
  legacy cos² regression coverage; production default is `power=1`
  (constant grid Re).
- `test_dca_extended_mse_conservation` — DCA closes STANDARD MSE
  `c_pd·∫dT + L_v·∫dq_v ~ 0` (latent heat released in-scheme via
  `delta_T_lh`); test was checking the older "no in-scheme latent heating"
  form.

### Regression tests (iter-8)
- `tests/ocean/unit/test_ocean.py::TestVerticalMixing::test_vertical_diffusion_cfl_cap_keeps_step_stable`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::test_sedimentation_cfl_positivity`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::test_sedimentation_surface_flux_conservation`

### Test status (post iter-19)
- 7 pre-existing test failures **all resolved** across iter-13 +
  iter-14: Stefan-Boltzmann (1 test, iter-13), hybrid_tracer path
  (1 test), ah_lat_scaling (4 tests), DCA MSE (1 test) — 6 fixed in
  iter-14 plus the iter-13 fix.  (Total = 1 + 1 + 4 + 1 = 7 tests
  across 4 categories.)
- 17 / 17 atmosphere hydrostatic integration tests pass.
- 169 / 169 atmosphere convection + microphysics + ocean vertical mixing
  targeted tests pass.
- 48 / 48 ocean implicit + MPAS + physics_ocean tests pass.
- 95 / 95 atmosphere turbulence tests pass.
- 10 / 10 sea-ice thermodynamics + 63 sea-ice dynamics/diff/transport tests
  pass.

## Inspected & clean (no fix needed)
- `land/snow_budget.py` — energy-limited melt with `constants.L_f` /
  `constants.T_freeze`; positivity guards intact.
- `land/carbon/carbon_cycle.py` — proper `jnp.maximum`/`jnp.clip` on GPP,
  phenology, NPP allocation.
- `land/richards.py` — implicit Picard + tridiagonal solve (unconditionally
  stable).
- `land/soil_thermal.py` — backward Euler + tridiagonal solve.
- `atmosphere/physics/turbulence/vertical_diffusion.py` — implicit (uses
  tridiagonal).
- `atmosphere/physics/clouds/cloud_fraction.py` — shared
  `saturation_mixing_ratio`, no inline Tetens.
- `atmosphere/physics/gravity_wave_drag/lindzen.py` — `constants.g`,
  `constants.p_ref`, `constants.kappa` references; differentiable scan.
- `ocean/physics/shortwave_penetration.py` — conservation correction at
  bottom layer; no flux leakage.
- `ocean/physics/bottom_drag/quadratic.py` — proper sign convention,
  shared dz-from-jacobian.
- `ocean/physics/lateral_mixing/gm_redi.py` — slopes are DM95-tapered to
  S_max; threading dt for a proper bolus-Courant cap would require a
  cross-path closure-signature change (deferred).
- `ice/rheology.evp_stress_update` — backward-Euler relaxation toward VP
  target; stable by construction.
- `ice/dynamics.evp_solver` — 120 subcycles + semi-implicit Coriolis (CICE
  convention).
- `atmosphere/physics/convection/dca.py` — moist static energy conservation
  enforced via the `delta_T_lh` block.
- `atmosphere/physics/microphysics/seifert_beheng.py` — proper joint donor
  clamps on q_c sinks; conservation enforced.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu}.py`
  — all use shared `virtual_temperature`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter (codex narrow review #5).
- Threading `dt` into `physics_fn` across all 3 ocean PE backends + 3 model
  drivers — substantial cross-file refactor; production runs use implicit
  vertical mixing (unconditionally stable) so the marginal stability gain
  is low.
- Visual / long-run validation of the new ice → ocean feedbacks.

## Next iterations
Continue addressing further codex adversarial-review findings until physics
implementation is provably conservative, monotone, and CFL-safe across
all parameterizations.

### Iteration 20 — 2026-05-12

Compression iteration.  Consolidated iter-10 through iter-19 entries into
the "Iterations 1-19 — Summary" section above so the working log stays
under the auto-loaded MEMORY.md / context envelope.  All previous detailed
entries remain in the commit messages on `clean_physics`.

### Iteration 22 — 2026-05-12

**Action: Sundqvist microphysics autoconversion donor clamp.**

Inspection found that
`atmosphere/physics/microphysics/sundqvist.py:diagnose_sundqvist_process_rates`
computes `P_auto = config.auto_rate · jnp.maximum(q_c + condensation·dt, 0)`
[kg/kg/s].  With the default `auto_rate = 1e-3` /s and a typical
physics step `dt = 1800 s`, the autoconversion sink times dt is
``auto_rate · dt = 1.8`` × the available cloud water — explicit Euler
update `q_c + dt · dq_c_dt` would drive q_c negative.

Fix: cap `P_auto = min(auto_rate · q_c_avail, q_c_avail / dt)` so
the per-step removal cannot exceed the local mass.  Added a
regression test that exercises an `auto_rate · dt = 1.8` configuration
and asserts `q_c_new ≥ 0`.

**Tests (post iter-22):**
- 7 / 7 Sundqvist + autoconversion regression tests pass.
