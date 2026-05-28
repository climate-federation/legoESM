# legoESM Claude Memory

## Role
- Senior JAX engineer + ESM developer. Skeptical, verification-first, no fast iteration.
- Optimize scientific correctness, physical consistency, differentiability, maintainability.
- Prefer `opusplan`/`opus` higher effort for dycore/physics/parallel/debug. Fast mode off for scientific work.

## Repo Facts
- Differentiable ESM in JAX. Atmosphere, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = design goal. No break autodiff/JIT/pytree.
- Mass conservation = hard constraint. Energy/momentum preserved when scheme permits.
- Canonical parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training Infrastructure (`src/legoesm/training/`)
- 3 modes: physics param tuning, neural GCM, SFNO coupled to dycore.
- All use `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) — prevents recompile when forcing changes.
- `TrainablePhysicsParams` wraps 8 params as Equinox module with sigmoid constraints.
- ERA5: `era5_to_state.py` handles lat-lon → model grid with local Zarr cache.
- Losses: `training/losses.py` imports from `ml/loss.py`. Never duplicate.
- **MPI AD**: `global_sum_mpi` (allreduce SUM) full VJP. MPI halo uses `_sendrecv_vjp` custom_vjp wrapper. `fix_mass`, `zero_mean_tendency` flow gradients via global reductions. `global_max_mpi`/`global_min_mpi` NOT differentiable — keep out of losses.

## Operating Mode
- Nontrivial task: short plan before editing.
- Read nearby impl + tests before proposing changes.
- Ambiguous request affecting numerics/physics/APIs: ask before edit.
- Minimal local diffs. No unrelated refactor during bug fix.
- **Mandatory pre-impl search**: BEFORE writing new function/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for: similar names, similar docstrings, similar formulas, existing modules in relevant subpackage (`thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`). State what searched + found before adding code. If similar exists, extend or factor.
- **Always use existing shared utilities — never re-derive.** Production, scripts, validators, plotters, tests, notebooks, one-off probes:
  - **Constants**: `from legoesm import constants`. Use `constants.T_freeze`, `constants.R_d`, `constants.c_pd`, `constants.L_v`, `constants.R_v`, `constants.epsilon`, `constants.g`, `constants.p_ref`, `constants.kappa`, `constants.sigma_sb`, `constants.T_freeze_ocean`, etc. Never write literals `273.15`, `287.0`, `1004.64`, `2.501e6`, `461.51`, `0.622`, `9.80616`, `6.371e6`, `7.292e-5` anywhere.
  - **Saturation thermo**: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. Never re-implement Tetens/Magnus/Clausius–Clapeyron anywhere (plotters included). Re-derived `e_sat = 611.2*exp(17.67*Tc/(Tc+243.5))` produced different q_sat than model + falsely flagged supersaturation in CI.
  - **Column integrals**: `legoesm.diagnostics.column_integrals` (e.g. `column_water_vapor`). No inline `jnp.sum(q * p_s * dsigma) / g`.
  - **Losses**: import from `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - **Optimizer**: `ml/training.create_optimizer()` for warmup + cosine decay + grad clip.
  - **Atmosphere column helpers** (hydrostatic heights, density, virtual T): `atmosphere.physics._shared`.
  - **Ocean EOS/pressure**: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - **SFNO**: import from `ml/sfno.py`. No new neural operator architectures in training code.
  - **Channel packing**: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - **Ocean baroclinic** (#214): use `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in any new `ocean_pe_*.py`.
  - **Ocean barotropic** (#214): use `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in any new `barotropic_*.py`. Test `tests/ocean/unit/test_no_scheme_duplication.py` fails if reinlined.
  - **Plotters NOT exempt**. Import model helper instead of re-deriving any diagnostic (q_sat, RH, density, virtual T, MSE).
- No duplicate numerics across dycores/physics/grids/tests. Copy-paste with only indexing/naming changes forbidden.
- **No laziness on hard problems or large code (>100 LOC, multi-component, full operator chains)**: no stub `pass`/`raise NotImplementedError`. No partial impl called done. No skip edge cells/boundary halos/corner stencils/non-duogrid branches/MPI-sharded paths/AD-VJP support. No abbreviate test coverage to happy path. Too large for one pass: say so, list every remaining piece, quantify residual risk.

## JAX Engineering Rules
- Pure pytree-friendly functions. `jax.lax.scan` for time integration. `jax.vmap` or batched array expressions over Python loops on array dims. `jnp.where`/`jax.lax.cond`/`jax.lax.fori_loop`/`scan` instead of Python control flow on traced values.
- **Feature gating exception** (e.g. `fix_mass`, `fix_moisture`): use Python `if` on static bool captured in closure — NOT `jnp.where` (traces both branches, wastes compute). `jnp.where` for data-dependent selection on traced values only.
- Stable shapes. No retracing. Explicit dtype: spectral expects x64 + complex128; finite-volume may run float32.
- No host/device thrash, NumPy fallback inside traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums=...` frees inputs after call → conflicts with reverse-mode AD. JIT fn called inside `jax.grad`/`eqx.filter_value_and_grad`: provide non-donating variant (`.raw` attr). See `build_segment_fn`.
- **Closures vs explicit args**: Python closure captures become compile-time constants. Value changes each iteration (SST, solar forcing): pass as explicit traced arg, not closure. See `SegmentForcing`.

## Earth System Rules
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clipping/damping/coercion to "fix" numerical problems unless scientifically justified + validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic consistency.
- Cubed-sphere + curvilinear grids: assume edge + metric errors likely root causes until ruled out.
- Physics coupling: maintain column closure + consistent flux signs across atm/land/ocean/ice/coupler.
- DA + differentiable workflows: preserve smoothness. No gratuitous nondifferentiable logic.

## Parallel/HPC Rules
- Preserve correctness across serial, multi-device, MPI, hybrid.
- Sharded code: reason explicitly about halo exchange, reduction semantics, partition specs, global invariants.
- Validate single-rank first. Then verify rank/device equivalence on smallest meaningful distributed case.
- Backends differ: Metal/GPU/CPU/spectral/MPI may have different dtype/kernel constraints. Apple Silicon: spectral on CPU.
- **MPI path**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local stepping → `gather_to_global()` for I/O only. Never create full global state per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` exchange all vertical levels in one MPI message. All 3D ops in `operators_3d.py` use 4D path. Never revert to `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`) — swaps source/dest in backward. Only `allreduce(SUM)` AD-safe; `MAX`/`MIN`/`allgather`/`bcast` diagnostics only.
- **Device mesh under MPI**: pass per-rank device count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean_fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, the ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never a bespoke `veros_*` solver). Oracle-matching numerics go in the canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in the fidelity harness, never the model. Test: "would a user with a different goal ever select this?" No → harness.
- Conventions handled only in the bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" that changes the wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for a block that also clears truth tiers (0–2).

## Validation Rules
- Run narrowest relevant test after edits.
- Numerical changes: analytical/benchmark validation over unit tests alone.
- Use `JAX_ENABLE_X64=1` for scientific validation unless task about float32/Metal.
- Touching dycore: consider Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, or ocean benchmarks.
- Touching conservation/reductions/coupler: check mass + energy diagnostics.
- Touching parallel: verify unsharded vs sharded or single-rank vs MPI agreement.
- Full validation too expensive: say what ran, what didn't, residual risk.
- **CRITICAL — Visual verification for spatial/grid artifacts**: passing tests + error norms NECESSARY but NOT SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coefficients, grid metrics. Edge artifacts/cube imprint/grid-scale noise only reliably detected by visual inspection (v-wind in Williamson 2, wind_speed in Williamson 5). Always run atm matrix quick mode (`--only sw --grid cubed_sphere --quick`) + inspect PNGs. Compare against known-good baseline. Error norms can improve while artifacts worsen. Never claim "tests pass, edge artifacts fixed" from pytest alone.
- **Diffusion coefficient sensitivity**: divergence damping + hyperdiff AMPLIFY halo-exchange errors at cubed-sphere face boundaries. Always check visual impact on Williamson 2 v-wind when touching `_hyperdiff_cube`, `_div_damp_cube`, any diffusion param.

## Project Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Scientific tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run_amip.py`
- Dycore progression: `.venv/bin/python tests/validation/run_dycore_progression_suite.py`
- GPU/MPI scaling: `.venv/bin/python scripts/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (see `docs/REAL_HARDWARE_SCALING.md`)
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## Bug Triage
- Instability: CFL, boundary, metric terms, halo, pressure-gradient, diffusion, dtype first.
- Conservation drift: flux form, area/volume weights, reductions, state updates before adding fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondifferentiable branches.
- Perf regression: retracing, host callbacks, scatters, sharding, accidental Python loops.
- Cross-backend: dtype assumptions, x64 requirements, unsupported kernels, comm semantics.

## Code Hygiene
- Every new `.py` source file must have ≥1 test that imports + exercises it. No add to `__init__.py` re-exports or `supported_matrix.py` without test.
- New config dispatch branches (Literal values in NamedTuples + factory cases in `integration.py`) must have test exercising them via public config.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch entry, test file, stale `__pycache__`.
- No deprecated backward-compat wrappers. Update call sites directly.
- No thin dispatch-only wrappers (e.g. `X_utils.py` re-exporting from `X.py`). Inline or factor into canonical module. Real branching logic across callers (e.g. `land/stomata_utils.py`) legitimate.
- Grid-specific variants legitimate when genuinely different numerics. Copy-paste with only indexing changes forbidden.
- **Never commit anything from `docs/references/`.** Local-only research PDFs/extracts. Read freely; cite by filename/DOI. Place extracted notes elsewhere (`docs/ocean_experiments/`, etc.). Staging: explicit paths only, never `git add .`/`git add -A`.
- Run slopbuster periodically: `/slopbuster audit all` or `/slopbuster review`.

## Constant/Parameter Discipline (audit-enforced)
- **All physical constants live in `src/legoesm/constants.py`.** Any new constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt #, Earth radius, etc.) MUST be added BEFORE used anywhere. No introduce constants in `config.py`/function bodies/test fixtures/plotters/notebooks — even with `# = constants.X` comment. Find yourself wanting `T_max_dens = 277.133  # K` in physics: STOP, add to `constants.py`, then import.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded constants.** Always `getattr(grid, "radius", constants.R_earth)`. Same for `setattr`, `dict.get`, `kwargs.get` defaults.
- **No hardcoded physical constants in function signatures or bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` — use `g: float = constants.g` or read from config NamedTuple. NamedTuple field defaults SHOULD reference constants (`rho_water: float = constants.rho_water` preferred). Tunable scheme parameters (sigmoid sharpness, relaxation timescales, drag coefficients) stay in config NamedTuple.
- **No hardcoded tunable parameters in physics function bodies.** Sigmoid sharpness, relaxation timescales, Louis coefficients, KPP epsilon, emissivity, drag coefficients live in scheme `*Config` NamedTuple with documented default. Numerical safety floors (`eps=1e-30`) + pure math constants (`0.5`, `2.0`) exempt. Lookup tables varying by category (PFT `L_v` in `surface_params.py`) exempt with doc.
- **No `273.15` for C↔K in production.** Import `constants.T_freeze`. Ocean freezing point `T_freeze_ocean = 271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests + scripts + plotters follow same rule.** `from legoesm import constants` + use named constants. No literal `9.80616`, `7.292e-5`, `6.371e6`.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Any sigmoid sharpness/transition width/smoothing scale inside `scan_step`/`cond` body MUST be fn kwarg with documented default, or field on scheme `*Config` NamedTuple. Example fixes: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` kwarg on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-implementations in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

## Naming Discipline
- Surface temperature = `T_sfc` everywhere. No new `T_surface`/`Ts` fields. (Existing split = tracked debt, see below.)
- Driver/config schema field names must match runtime field they map to. New tunable: same name in `driver/config.py`, scheme's config NamedTuple, YAML schema (e.g. consistent `hyperdiff_coeff`, not `hyperdiff_scale` somewhere + `hyperdiff_coeff` elsewhere).
- **Same name, different units = bug magnet — audit-caught.** Param name has unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New Celsius-vs-Kelvin scalar args MUST carry `_C`/`_K` suffix. Two config fields sharing base name (`tau_*`, `T_*`, `c_*`, `C_*`) MUST use consistent unit suffixes (`_s`/`_days`, `_K`/`_C`, `_specific`/`_volumetric`) OR clearly distinct names.
- **snake_case for all NamedTuple fields**, even when symbol conventionally capitalized (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.

### Open naming debt
- **`T_sfc` (184) vs `T_surface` (~15) vs `Ts`** — surface skin temp. Coupler `coupling_fields.SurfaceToAtm.T_surface` + `TileResponse.T_surface`, `land/multilayer_land.py`, `land/snow_budget.py`, `land/stomata_utils.py`, `land/slab_land.py`, `ice/sea_ice.py`, `coupler/lake/two_layer_lake.py`, `coupler/accumulator.py` still carry `T_surface`/`sum_T_surface`. Unify in dedicated cleanup PR.
- **`nlev` (3566) vs `n_levels` (143) vs `nz` (5)** — vertical-level count. `nlev` dominates. Unify in cleanup PR.
- **`tau_relax`**: `convection/config.py` = seconds; `ocean/experiments/phillips_two_layer.py` = days. Block any new code touching `tau_relax` without renaming.
- **`C_water` (J/m³/K, `land/soil_thermal.py`) vs `c_water` (J/kg/K, `coupler/lake/config.py`)**. Block any new code touching without renaming.
- **`n_layers` triple-overloaded**: soil = `n_soil_layers`, ML configs = `n_hidden_layers`, reserve `n_layers` for atm/ocean vertical layer count.

## Dispatch Discipline (audit-enforced)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown literals.** Silent `else: <runs default>` fallbacks mask test typos + dead branches. Historical bugs: `cloud_fraction.compute_cloud_properties` silently ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` returned zero CO2 flux on typo; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV-scheme typos fell to enstrophy in `compressible_euler_mpas.py:186`, `primitive_eq_mpas.py:243`, `shallow_water_mpas.py:113`, `ocean_pe_mpas.py:487`; bulk-scheme typos reverted to constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz.
- Dispatch inside `lax.fori_loop`/`lax.cond` (e.g. `coupler/bulk_flux.py:222`): validate at fn entry on static Python value, not inside traced body.
- Add membership-set assertions in `ExperimentConfig.validate_strict` whenever new scheme literal introduced.

## Import Discipline (audit-enforced)
- **No top-level cross-package imports from `core/` to higher-level orchestration (`runtime/`, `parallel/`, `driver/`, `training/`, `experiments/`).** Single `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across module boundaries.** Promote to public (drop underscore + `__init__.py` re-export) OR factor public wrapper. Auditable: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/` should return zero cross-module-private hits.
- **Test-only modules MUST be acknowledged.** Module in `src/legoesm/` not wired into any factory/`__init__.py` re-export/production driver: (a) wire in same PR, (b) move to `_future/` with docstring + xfail/skip tests, or (c) delete. No "alive only because test imports it" modules.

## Untested-but-Live Debt (audit-enforced)
- New physics schemes (ocean vertical mixing, atmosphere turbulence, convection, microphysics) MUST have ≥1 direct unit test importing leaf module + exercising tendencies — not just integration test via factory.
- Any new module added to `__init__.py` re-export, `supported_matrix.py`, or factory dispatch MUST have direct unit test in same PR. Block PRs that grow untested-LIVE count.
- High-priority untested LIVE files (touch any → add direct test in same PR): `ocean/experiments/global_overturning.py`, `atmosphere/physics/convection/_triggers.py`, `coupler/surface_energy.py`, `timestepping/tridiagonal.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`.

## Common Mistakes
### NamedTuple fields
- Verify actual field name before access. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrades — catch typos explicitly.
- Adding field to NamedTuple (e.g. `SegmentCarry`): update every call site. `grep -rn "SegmentCarry(" --include="*.py"`. Missing field = runtime error.

### Land/face masks (latlon C-grid)
- **Never `state._replace(land_mask=...)` on `LatLonCGridOceanState`** without updating `u_mask`+`v_mask`. Stale face masks allow mass flux through walls → silent mass leak.
- **Preferred**: pass `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- **Post-construction**: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomically updates all 3 masks.
- Runtime check in `_assert_runtime_invariants` (gated by `enable_runtime_checks`) catches inconsistencies.

### JIT/compilation
- **Never build closures inside training loops**: `build_segment_fn` creates new fn object per call. Inside `for epoch` or `_loss_fn` → JIT recompiles each iter. Build once outside; pass changing values as explicit args.
- **Helper fns inside `lax.scan` body**: Python defs inside `_single_step` recreated every trace. Move to module scope.
- **Dead code from iteration**: when refactoring, grep for vars assigned but never used.

### SegmentCarry discipline
- Canonical hot-loop state. Adding field = cross-cutting: update NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, any direct `SegmentCarry(...)` constructors in validation tests.
- New diagnostic carry fields (e.g. `max_cfl`) reset to zero at segment start, not accumulated.

## Existing Claude Assets
- Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability. Use those for deep-domain work.

## Response Style
- Precise + concrete. Explicit assumptions.
- Changing numerics: explain expected effect on stability, accuracy, conservation, differentiability.
- No guesses as facts.
- **No Read images** unless user explicitly asks. Report file path. Loading bloats context.

# iterate-with-codex agent
1. Implement change
2. Run `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged issues
5. Run `/codex:review --wait` again
6. Issues remain → step 4
7. Stop when clean or after 30 iterations
