# legoESM Claude Memory

## Role
- Act like senior JAX engineer + Earth system model developer.
- Default careful, skeptical, verification-first. No fast iteration.
- Optimize for scientific correctness, physical consistency, differentiability, maintainability.
- If model selection available, prefer `opusplan` or `opus` higher effort for nontrivial dycore, physics, parallel, debug work.
- Fast mode off for scientific impl, numerical debug, architecture changes unless user prioritizes latency.

## Repo Facts
- Differentiable Earth system model in JAX. Spans atmosphere, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = design goal. No break autodiff, JIT, pytree semantics for convenience.
- Conservation matters: mass = hard constraint. Energy/momentum consistency preserved when scheme permits.
- Canonical parallel entry: `ParallelRuntime.create()`.
- Supports cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral pathways.

## Training Infrastructure (`src/legoesm/training/`)
- **Three modes**: physics param tuning, neural GCM, SFNO coupled to dycore.
- All modes use `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad` for AD compat.
- `SegmentForcing` = explicit arg to `run_segment`, not closure-captured — prevents recompile when forcing changes.
- `TrainablePhysicsParams` wraps 8 physics params as Equinox module with sigmoid constraints.
- ERA5 data: `era5_to_state.py` handles lat-lon → model grid with local Zarr cache.
- Losses: `training/losses.py` imports from `ml/loss.py` — never duplicate loss functions.
- **MPI AD compat**: `global_sum_mpi` (allreduce SUM) full VJP support. MPI halo exchange uses `_sendrecv_vjp` custom_vjp wrapper. Conservation fixers (`fix_mass`, `zero_mean_tendency`) flow gradients correctly through global reductions. `global_max_mpi` / `global_min_mpi` NOT differentiable — keep out of loss functions.

## Operating Mode
- Nontrivial task: start with short plan before editing.
- Read nearby impl + tests before proposing or making changes.
- Request ambiguous + could affect numerics/physics/APIs/scientific conclusions: ask clarifying question before edit.
- Prefer minimal, local diffs. No refactor unrelated code during targeted bug fixes.
- **Mandatory pre-impl search**: BEFORE writing any new function, helper, class, operator, diagnostic, init/load path, loss, numerical routine, MUST search codebase for existing impl. Use Grep/Glob/Explore across `src/legoesm/` for: similar function names, similar docstrings, similar formulas, existing modules in relevant subpackage (e.g., `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`). State explicitly what searched + found before adding new code. If something similar exists, extend or factor — no duplicate.
- **Always use existing shared utilities by default — never re-derive.** Default for *every* task: production source, scripts, validation harnesses, plotting code, tests, notebooks, one-off probes. Specifically:
  - **Constants**: `from legoesm import constants` — `constants.T_freeze`, `constants.R_d`, `constants.c_pd`, `constants.L_v`, `constants.R_v`, `constants.epsilon`, `constants.g`, `constants.p_ref`, `constants.kappa`, `constants.sigma_sb`, `constants.T_freeze_ocean`, etc. Never write literals like `273.15`, `287.0`, `1004.64`, `2.501e6`, `461.51`, `0.622`, `9.80616` in any file (production, scripts, tests, plotters). Audit-enforced for `src/legoesm/`; same rule extends to `scripts/` and `tests/`.
  - **Saturation thermodynamics**: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice` — never re-implement Tetens / Magnus / Clausius–Clapeyron in any file (plotters included). Plotter re-derived `e_sat = 611.2*exp(17.67*Tc/(Tc+243.5))` produced *different* q_sat than model's saturation adjustment + falsely flagged supersaturation in CI check. Rule exists because formula disagreement masquerades as physics bug.
  - **Column integrals**: use `legoesm.diagnostics.column_integrals` (e.g. `column_water_vapor`).
  - **Loss functions**: import from `ml/loss.py`.
  - **Optimizer setup**: `ml/training.create_optimizer()`.
  - **Atmosphere column helpers** (hydrostatic heights, density, virtual temperature): `atmosphere.physics._shared`.
  - **Ocean EOS / pressure**: `ocean.eos`.
  - **Plotting** of model output: when computing diagnostic quantities (q_sat, RH, density, virtual T, MSE, etc.) inside plotter, *import* model's helper instead of re-deriving. Plotters NOT exempt from "no re-derivation" rule.
  When in doubt, search `src/legoesm/` for function before writing one.
- No duplicate numerics across dycores, physics packages, grids, test setups when common impl feasible. Copy-paste with only indexing/naming changes forbidden.
- No trade correctness for speed by skipping validation or speculative edits.
- **No laziness on hard problems or large code production**: challenging task (numerical bug hunts, dycore ports, multi-file refactors, new parameterizations) or substantial code (>100 LOC, multi-component changes, full operator chains, large test matrices) → MUST do full work. No stub functions with `pass` or `raise NotImplementedError`. No partial impl called done. No skip harder corner cases (edge cells, boundary halos, corner stencils, non-duogrid branches, MPI/sharded paths, AD/VJP support) + quietly leave for later. No abbreviate test coverage to single happy path. Task genuinely too large for one pass: say so explicitly, list every remaining piece, quantify residual risk — never imply completion not delivered.

## JAX Engineering Rules
- Keep functions pure + pytree-friendly.
- Prefer `jax.lax.scan` for time integration + structured loops.
- Prefer `jax.vmap` or batched array expressions over Python loops on array dims.
- Use `jnp.where`, `jax.lax.cond`, `jax.lax.fori_loop`, `scan` instead of Python control flow on traced values.
- **But**: on/off feature gating (e.g., `fix_mass`, `fix_moisture`) use Python `if` on static bool captured in closure — NOT `jnp.where`, which traces both branches + wastes compute. `jnp.where` for data-dependent selection on traced values only.
- Preserve stable shapes. No unnecessary retracing.
- Explicit dtype behavior. Spectral solvers expect x64 + complex128. Finite-volume pathways may intentionally run float32.
- No host/device thrash, unnecessary materialization, ad hoc NumPy fallbacks inside traced code.
- No hidden non-JAX side effects that break JIT, grad, checkpointing, sharding.
- **Buffer donation + `jax.grad`**: `@jax.jit(donate_argnums=...)` frees input buffers after call. Conflicts with reverse-mode AD, needs inputs for backward. JIT-compiled function called inside `jax.grad` or `eqx.filter_value_and_grad`: provide non-donating variant (e.g., `.raw` attr) + use for training. See `build_segment_fn` for pattern.
- **Closures vs explicit args for JIT reuse**: values captured in Python closure become compile-time constants. Value changes every iteration (e.g., SST, solar forcing): pass as explicit traced argument — not closure capture — so compiled kernel reused. See `SegmentForcing` for pattern.

## Earth System Modeling Rules
- Treat conservation, metric consistency, staggered-grid consistency, halo correctness as first-class requirements.
- Never "fix" numerical problems with silent clipping, damping, coercion unless scientifically justified + validated.
- Preserve units, sign conventions, monotonicity/positivity assumptions, hydrostatic/nonhydrostatic consistency.
- Cubed-sphere + other curvilinear grids: assume edge + metric errors likely root causes until ruled out.
- Physics coupling: maintain column closure + physically consistent flux signs across atmosphere, land, ocean, ice, coupler interfaces.
- DA + differentiable workflows: preserve smoothness where intended. No gratuitous nondifferentiable logic.

## Parallel and HPC Rules
- Preserve correctness under serial, multi-device, MPI, hybrid execution.
- Sharded/distributed code: reason explicitly about halo exchange, reduction semantics, partition specs, global invariants.
- Validate single-rank behavior before assuming distributed bug fixed.
- Then verify rank/device equivalence on smallest meaningful distributed case.
- No assume Metal, GPU, CPU, spectral, MPI backends have identical dtype or kernel constraints.
- Apple Silicon: keep spectral work on CPU.
- **MPI distributed path**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local stepping → `gather_to_global()` for I/O only. Both ModelDriver + benchmark script use this path. Never create full global state per rank — always scatter.
- **Native 4D halo exchange**: `pad_halo_4d()` + `pad_halo_vector_4d()` exchange all vertical levels in one MPI message. All 3D operators in `operators_3d.py` use 4D path. No revert to `vmap(pad_halo)` which issues `nlev` separate messages.
- **MPI halo AD safety**: All `sendrecv` calls go through `_sendrecv_vjp` (`@jax.custom_vjp` wrapper in `halo_exchange.py`) that swaps source/dest in backward. Enables `jax.grad` through MPI halo exchange. Only `allreduce(SUM)` AD-safe among MPI reductions. `MAX`, `MIN`, `allgather`, `bcast` not differentiable — use only in diagnostics.
- **Device mesh under MPI**: Pass per-rank device count to `create_device_mesh()`, not total across all ranks. Function warns when clamping.

## Validation Rules
- Always run narrowest relevant test after edits.
- Numerical changes: prefer analytical or benchmark-style validation over unit tests alone.
- Use `JAX_ENABLE_X64=1` for most scientific validation unless task specifically about float32 or Metal portability.
- Touching dycore operators: consider Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, or equivalent ocean benchmarks as appropriate.
- Touching conservation, reductions, coupler logic: check mass + energy diagnostics explicitly.
- Touching parallel code: verify unsharded vs sharded or single-rank vs MPI agreement.
- Full validation too expensive: say exactly what ran, what not ran, residual risk.
- **CRITICAL — Visual verification for spatial/grid artifacts**: Passing unit tests + error norms NECESSARY but NOT SUFFICIENT when modifying cubed-sphere operators, halo exchange, diffusion coefficients, grid metrics. Edge artifacts, cube imprint, grid-scale noise ONLY reliably detected by visual inspection of field snapshots (especially v-wind in Williamson 2, wind_speed in Williamson 5). Always run atmosphere test matrix quick mode (`--only sw --grid cubed_sphere --quick`) + inspect generated snapshot PNGs before claiming fix works. Compare against known-good baseline image. Error norms can improve while visual artifacts get worse (e.g., artifacts shift location or change character). Never claim "tests pass, edge artifacts fixed" based on pytest results alone.
- **Diffusion coefficient sensitivity**: Divergence damping + hyperdiffusion coefficients AMPLIFY halo-exchange gradient errors at cubed-sphere face boundaries. Increasing coefficients (even modestly) can worsen edge artifacts. Always check visual impact on Williamson 2 v-wind when changing `_hyperdiff_cube`, `_div_damp_cube`, or any diffusion parameter.

## Project-Specific Commands
- Install: `pip install -e ".[dev]"`
- General tests: `.venv/bin/python -m pytest tests/`
- Targeted scientific tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atmosphere test matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py`
- Ocean test matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py`
- AMIP production: `.venv/bin/python scripts/run_amip.py`
- Dycore progression suite: `.venv/bin/python tests/validation/run_dycore_progression_suite.py`
- GPU/MPI scaling benchmark: `.venv/bin/python scripts/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (see `docs/REAL_HARDWARE_SCALING.md`)
- MPI distributed tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI differentiability tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## How To Think About Bugs
- Instability: check CFL, boundary treatment, metric terms, halo exchange, pressure-gradient formulation, diffusion, dtype first.
- Conservation drift: inspect flux form, area/volume weights, reductions, state updates before adding fixers.
- Differentiability failures: inspect control flow, shape changes, side effects, checkpointing, nondifferentiable branches.
- Performance regressions: look for retracing, host callbacks, excessive scatters, poor sharding, accidental Python loops.
- Cross-backend discrepancies: inspect dtype assumptions, x64 requirements, unsupported kernels, communication semantics.

## Code Hygiene Rules
- Every new `.py` source file must have at least one test that imports + exercises it. No add files to `__init__.py` lazy imports or `supported_matrix.py` without corresponding test.
- New config dispatch branches (new Literal values in config NamedTuples + factory cases in `integration.py`) must have test exercising that branch.
- Every dispatch branch must have at least one test selecting it via public config (not just unit test of leaf module).
- Removing source module: also remove its `__init__.py` re-export, its `supported_matrix.py` entry, its dispatch entry, its test file, any stale `__pycache__` files.
- No deprecated backward-compat wrappers. API changes: update call sites directly.
- No thin dispatch-only wrappers (e.g., `X_utils.py` that just re-exports function from `X.py`). Inline call at each site or factor helper into canonical module. (Modules with real branching/dispatch logic across multiple callers — like `land/stomata_utils.py` — legitimate, not "thin wrappers".)
- Grid-specific variants legitimate when genuinely different numerics. Copy-paste with only indexing changes forbidden — factor shared logic into common function.
- **Never commit anything from `docs/references/`.** Directory = local-only stash for research PDFs (papers, technical reports, multi-MB figures) consulted while planning. Must not enter git history under any circumstances — not PDFs themselves, not extracts, not derivative summary files placed in that folder. Read freely. Cite by filename or DOI in commit messages + design docs. Place extracted notes elsewhere (`docs/ocean_experiments/`, etc.). Staging files: use explicit paths — never `git add .` or `git add -A` in this repo, since `docs/references/` not currently enforced by `.gitignore`.
- Run slopbuster agent (`/slopbuster audit all` or `/slopbuster review`) periodically, especially before releases.

## Constant and Parameter Discipline (audit-enforced)
- **All physical constants live in `src/legoesm/constants.py`. PERIOD.** Any new physical constant — temperature, density, specific heat, latent heat, thermal conductivity, viscosity, EOS coefficient, Kell density-curvature, Schmidt number, Earth radius, etc. — MUST be added to `legoesm.constants` BEFORE used anywhere else. Do NOT introduce new physical constants directly in `config.py` files, function bodies, test fixtures, plotting scripts, notebook cells — even with `# = constants.X` annotation comment. Rule: if you find yourself wanting to write `T_max_dens = 277.133  # K` inside physics module, STOP, add `T_freshwater_max_density = 277.133` to `constants.py`, then `from legoesm import constants` + reference `constants.T_freshwater_max_density`. Applies whether writing production code in `src/legoesm/`, scheme `config.py`, test, plotter, one-off probe.
- **`getattr(..., "radius", <literal>)` fallbacks count as hardcoded constants.** Pattern like `R = getattr(grid, "radius", 6.371e6)` ships hardcoded `6.371e6` default that drifts from `constants.R_earth = 6.371229e6`. Always use `getattr(grid, "radius", constants.R_earth)` (or import the specific constant) so fallback path stays in sync with canonical value. Same rule applies to `setattr` defaults, `dict.get` defaults, `kwargs.get(..., <literal>)`.
- **No hardcoded physical constants in function signatures or function bodies in production code (`src/legoesm/`).** Function defaults like `def f(g=9.80616, ...)` forbidden — use `from legoesm import constants` + `g: float = constants.g`, or read from config NamedTuple. Includes ocean experiment scaffolding: module-level patterns like `_G_EARTH = 9.80616` must be replaced with `constants.g`. **NamedTuple field defaults that duplicate physical constant SHOULD reference constants module** (`rho_water: float = constants.rho_water` preferred over `rho_water: float = 1000.0  # = constants.rho_water`). Existing NamedTuples with annotated literal defaults grandfathered, but new fields must use `constants.X` reference. Truly tunable scheme parameters (sigmoid sharpness, relaxation timescales, drag coefficients) stay in their config NamedTuple — those = scheme tunings, not physical constants.
- **No hardcoded tunable parameters in physics function bodies.** Sigmoid sharpness, relaxation timescales, Louis coefficients, KPP epsilon, surface emissivity, drag coefficients, etc. must live in scheme's config NamedTuple with documented default. Numerical safety floors (`eps=1e-30` for division) + pure mathematical constants (`0.5` in midpoint averaging, `2.0` in squared norms) exempt. Lookup tables that legitimately vary constant by category (e.g., PFT-dependent `L_v` rounding in `surface_params.py`) exempt — but document why.
- **No `273.15` for Celsius↔Kelvin conversions in production code.** Import `constants.T_freeze`. The ocean freezing point (271.35 K for seawater) is intentionally different from freshwater `T_freeze = 273.15 K` in `ocean/eos.py` = only intentional exception — stays literal with comment.

## Naming Discipline
- Surface temperature = `T_sfc` everywhere — atmosphere, land, coupler, diagnostics, ocean. No new `T_surface` or `Ts` fields. (Existing split = tracked tech debt; see "Open naming debt" below.)
- Driver/config schema field names must match runtime field they map to. New tunable: use same name in `driver/config.py`, scheme's config NamedTuple, any YAML schema (e.g., consistent `hyperdiff_coeff`, not `hyperdiff_scale` in one + `hyperdiff_coeff` in another).
- **Same name, different units = bug magnet — now caught by audit.** Parameter name has physical-unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`, etc.): use everywhere quantity appears. Slopbuster audit 2026-05-13 renamed ocean-init Celsius initial-condition arg from `T_surface: float = 20.0 (°C)` to `T_water_init_C` to disambiguate against atmospheric/coupler/land `T_sfc` (or surviving `T_surface`) Kelvin tensor field. Future Celsius-vs-Kelvin scalar args MUST carry `_C` / `_K` suffix.

### Open naming debt
- **`T_sfc` (184 sites) vs `T_surface` (~15 sites in coupler/land Kelvin tensors)** — surface-skin-temperature field. Coupler `coupling_fields.SurfaceToAtm.T_surface` + `TileResponse.T_surface` (Kelvin pytree leaf) plus `land/multilayer_land.py`, `land/snow_budget.py`, `land/stomata_utils.py`, `land/slab_land.py`, `ice/sea_ice.py`, `coupler/lake/two_layer_lake.py`, `coupler/accumulator.py` still carry `T_surface` / `sum_T_surface` field names. Unify to `T_sfc` in dedicated cleanup PR — not bundled into unrelated work.
- **`nlev` (3566 sites) vs `n_levels` (143) vs `nz` (5)** — vertical-level count uses three names. `nlev` dominates. Unify in dedicated cleanup PR.

## Untested-but-live debt (audit-enforced)
- New physics schemes (ocean vertical mixing, atmosphere turbulence, convection, microphysics) must have at least one direct unit test that imports leaf module + exercises its tendencies — not just integration test that hits it via factory. 2026-04-28 audit found 12 high-risk leaf modules in `ocean/physics/` + `atmosphere/physics/turbulence/` reachable only through `integration.py` with no direct test. No extend this pattern.

## Audit lessons — 2026-05-03 cycle
- **Eager top-level imports across packages cause silent CI fragility.** Single `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking every isolated `pytest tests/ocean/unit/test_X.py` invocation while bulk collection happened to paper over it. **Rule:** low-level package (`core/`) needs symbols from higher-level orchestration package (`runtime/`, `driver/`): defer import inside function/method that needs it. No add eager top-level cross-package imports that re-enter through `__init__.py`.
- **`scheme="..."` factories must `raise ValueError` for unknown literals.** `cloud_fraction.compute_cloud_properties` previously had `if scheme == "xu_randall": ... else: sundqvist`, so any unknown literal silently ran as sundqvist. Every dispatch site in `integration.py` already does `raise ValueError(f"Unknown ... scheme: {config.scheme!r}")`. cloud_fraction now does too. **Rule:** every public-config dispatch must validate scheme literal explicitly + raise on unknown values. Silent fallbacks mask test typos + dead branches. Add matching validation in `ExperimentConfig.validate_strict` whenever new scheme literal introduced.
- **Same field name, different units = bug magnet.** Audit found `tau_relax` meaning *seconds* in `convection/config.py` + *days* in `ocean/experiments/phillips_two_layer.py`. `c_water` (J/kg/K) in `lake/config.py` vs `C_water` (J/m³/K) in `land/soil_thermal.py`. `T_ice` used as freezing-point parameter in `driver/config.py`/`forcing/amip.py` when actually `T_freeze_ocean = 271.35 K`. **Rule:** two config fields share base name (`tau_*`, `T_*`, `c_*`, `C_*`): must use consistent unit suffixes (`_s` / `_days`, `_K` / `_C`, `_specific` / `_volumetric`) OR clearly distinct names. No let two NamedTuples carry same identifier with different physical meaning.
- **Hardcoded function defaults for physical constants forbidden in `src/legoesm/`.** Audit found ~25 sites with `def f(g=9.80616, ...)` or `def f(p_s_init: float = 1.0e5, ...)` style defaults that drift from `constants.g`/`constants.p_ref`. **Rule:** function signatures in `src/legoesm/` must use `g: float = constants.g` (preferred) or read value from config NamedTuple — never bare literal. NamedTuple field defaults may use literal floats with `# = constants.X` comment per existing exception.
- **Tests + scripts must follow constant-hygiene rule too.** 2026-05-03 audit found 80+ test sites + 30+ `scripts/diag_*.py` sites still hardcoding `g = 9.80616` + `omega = 7.292e-5`. **Rule:** new tests, plotters, diagnostic scripts must `from legoesm import constants` + use named constants. Use `constants.g`, never `9.80616`.
- **Test-only modules must be acknowledged as test-only.** Files (`core/fv3_d_sw5_corner_divergence.py`, `ocean/dynamics/ocean_pe_fc.py`, `core/_future/vertical_remap.py`, `ocean/diagnostics.py`) reachable *only* from own unit tests — not from any production driver, factory, `supported_matrix.py` entry — = tech debt. **Rule:** `src/legoesm/` module not wired into any factory dispatch, `__init__.py` re-export, or production driver: either (a) wire it in within same PR that adds it, (b) move to `_future/` subpackage with docstring stating staged-not-integrated + mark its tests `@pytest.mark.xfail` or `@pytest.mark.skip(reason="not yet wired")`, or (c) delete it. No "alive only because test imports it" modules.

## Audit lessons — 2026-05-13 cycle
- **`getattr(grid, "radius", 6.371e6)` fallbacks count as hardcoded constants.** Slopbuster Pass 9 (2026-05-13) found three sites (`ocean/dynamics/latlon_cgrid_operators.py`, `ocean/diagnostics_streamfunction.py` ×2) shipping hardcoded `6.371e6` Earth-radius fallback that drifts from `constants.R_earth = 6.371229e6`. **Rule:** fallback values in `getattr` / `setattr` / `dict.get` / `kwargs.get` MUST be `constants.X`, never literal. Same rule applies to NamedTuple field defaults.
- **Sigmoid sharpness / transition widths in hot loops must be configurable.** Slopbuster Pass 10 (2026-05-13) found `jax.nn.sigmoid((p - p_lcl)/100.0)` (LCL transition, `atmosphere/physics/thermodynamics.py`) + `jax.nn.sigmoid(delta_rho * 1e4)` (plume active mask, `ocean/physics/convection/plume.py`) hardcoded inside scan bodies. Both lifted: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)` + `PlumeConfig.active_sigmoid_sharpness=1e4`. **Rule:** any sigmoid sharpness, transition width, smoothing scale that lives inside `scan_step` / `cond` body MUST be function kwarg with documented default, or field on scheme's `*Config` NamedTuple. Hardcoded magic numbers in JAX hot loops forbidden.
- **`ml/` subtree = staged-not-integrated experimental scaffolding.** Slopbuster Pass 7 (2026-05-13) found 36 modules in `src/legoesm/ml/` have zero imports from non-`ml/` source — subtree fully self-contained. Publicly re-exported names (`SFNO`, `SFNOBlock`, `SpectralConv`, channel-packing helpers, conservation correctors) ARE covered by direct tests + safe to import from production callers. Rest of `ml/` (especially `ml/s2s/`, `ml/physics/`, `ml/training.py`) = research scaffolding. **Rule:** no extend untested-leaf footprint of `ml/`. New work either (a) lands with at least one direct unit test, or (b) goes to `_future/` sibling per staged-not-integrated convention. See `src/legoesm/ml/__init__.py` docstring for full audit status.
- **Direct-unit-test debt paid down for top hot-loop leaves.** New tests landed for `timestepping/{dispatch,integration,pytree_ops,split_explicit}.py` (`tests/unit/test_timestepping_leaves.py`) + `diagnostics/column_integrals.py` (`tests/unit/test_diagnostics_column_integrals.py`). **Rule:** touch any of these modules: extend corresponding direct-unit-test file in same PR. No rely on indirect coverage through dycore drivers.

## Audit lessons — 2026-05-23 cycle
- **Silent fallback dispatch is the #1 physics-correctness footgun.** Slopbuster 2026-05-23 found 10 dispatch sites with `if scheme == X ... elif scheme == Y ... else: <runs default impl>` and NO `raise ValueError`. A typo `scheme="differlnd"` silently returns zero CO2 flux from the carbon cycle (`land/carbon/carbon_cycle.py:443`). Ocean biogeochemistry typos silently disable BGC (`ocean/biogeochemistry/carbon_cycle.py:108,209`). MPAS PV-scheme typos silently fall to enstrophy in 3 dycores (`compressible_euler_mpas.py:186`, `primitive_eq_mpas.py:243`, `shallow_water_mpas.py:113`) + ocean PE (`ocean_pe_mpas.py:487`). Bulk-scheme typos silently revert to constant coefficients across 5 sites (`coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`). `io/restart.py:232` silently writes npz on unknown backend. **Rule:** every public-config dispatch site MUST end with `else: raise ValueError(f"Unknown {field} scheme: {scheme!r}")`. When dispatch lives inside `lax.fori_loop` / `lax.cond` (e.g., `coupler/bulk_flux.py:222`), validate at function entry on the static Python value, not inside the traced body. Add membership-set assertions in `ExperimentConfig.validate_strict` for the bulk_scheme literal (used by 5 configs) so a single typo gets caught at config-build time.
- **Private (`_`-prefixed) symbols leak across module boundaries when their public twin never exists.** Pass 11 (2026-05-23) found ~10 `_private` symbols imported cross-module without a public wrapper. Worst offenders: `core.operators._is_distributed` (4 callers), `core.operators_fv._ppm_edge_values` + `_ppm_limit` (5 callers across `advection.py`, `operators_fv_latlon.py`, `operators_fv_latlon_3d.py`), `ocean.conservation._ocean_global_sum` (5+ callers — **fixed in this cycle**: promoted to `ocean_global_sum`), `core.precision._resolve_dtype`, `grids.halo._interp_strip` / `_pad_halo_local` / `_pad_halo_local_4d` (used by `parallel/cubesphere_exchange.py`, `parallel/async_halo.py`), `parallel.reductions._require_mpi_stack`, `runtime.backend._detect_gpu_vendor_pre_init`, `physics_pipeline._get_turbulence_fn` / `_get_gwd_fn`. **Rule:** when a `_private` symbol is imported by ≥1 module outside its own file, EITHER promote it (drop underscore, add to `__init__.py` re-export) OR factor a public wrapper that delegates. Same-package-different-file imports do not justify keeping the underscore. Auditable check: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/` should return zero hits on truly cross-module-private symbols.
- **Cross-package top-level imports from `core/` to higher-level orchestration (`runtime/`, `parallel/`) re-introduced.** 2026-05-03 audit fixed this exact pattern; 2026-05-23 found it BACK at `core/__init__.py:6` (`from legoesm.runtime.backend import check_spectral_backend, get_backend`) and `core/conservation.py:34` (`from legoesm.runtime.backend import is_x64_enabled, supports_float64`). **Rule:** any new top-level import from `core/` to `runtime/`, `parallel/`, `driver/`, `training/`, or `experiments/` is FORBIDDEN. Use function-scope deferred imports. `core/operators_3d.py:34` also top-level imports `legoesm.parallel.async_halo` — same risk class, move to function scope.
- **Hardcoded `0.622`, `273.15`, `5.67e-8` in ocean coupler/forcing modules that already `from legoesm import constants`.** `ocean/coupler/omip2_applicator.py` has 5 Celsius→K conversions (`T_C = T_K - 273.15` ×3 at lines 194,256,303; 1 at line 53), one `0.622 * e_s / ...` saturation re-implementation (line 55), and one `getattr(constants, "sigma_sb", 5.67e-8)` fallback (line 187) that exactly violates the 2026-05-13 getattr rule. `ocean/forcing/jra55_do.py:111,114` re-implements Bolton saturation with literal `0.622` and `273.15`. **Rule:** when a file already imports `legoesm.constants`, ALL physical-constant literals matching a canonical value MUST resolve through `constants.X`. Saturation re-implementations in forcing modules are forbidden — use `legoesm.thermo.saturation_mixing_ratio` / `saturation_vapor_pressure`.
- **Sigmoid sharpness in scan bodies — second wave.** 2026-05-13 audit lifted 2 sites. 2026-05-23 found 11 more: `ysu.py:163` (`10.0` PBL blend, sister field exists on HBConfig), `edmf.py:229` (`20.0` in `_step` scan body), `louis.py:136` (`100.0` Ri-branch blend), `_triggers.py:321` (`20.0` shared utility used by every plume LFC/LCL/LNB), `bechtold.py:321,341` (`2.0` below-LCL + `10.0` downdraft RH), `tiedtke.py:296,309` (`2.0`+`10.0`, mirrors bechtold), `emanuel.py:220` (`2.0`), `dca.py:114` (`10.0` smooth trigger in `_step_pair`). **Rule:** convection + turbulence schemes ALL must expose their sigmoid sharpnesses / transition widths as scheme-config NamedTuple fields. The three `below_lcl_sigmoid_sharpness=2.0` sites in bechtold/tiedtke/emanuel should share a helper kwarg or each scheme's config gets the field. The `_triggers.smooth_lowest_crossing_index` helper MUST accept `gate_sharpness` kwarg.
- **DM95 GM/Redi taper transition-width `0.1·S_max` triplicated.** `ocean/physics/lateral_mixing/_gm_redi_common.py:52,73` and `gm_redi_latlon_cgrid.py:338` each hardcode `0.1` for `(S_max − |S|) / (0.1 · S_max + eps)`. **Rule:** lift to `transition_width_frac: float = 0.1` kwarg on `dm95_taper`, `dm95_taper_scalar`, `_triad_taper`, with matching field on `GMRediConfig`. One point of tuning instead of three.
- **Same name, different units recurrence: `tau_relax` and `C_water`/`c_water` still unfixed.** 2026-05-03 audit flagged both; 2026-05-23 confirms `ocean/experiments/phillips_two_layer.py:98` still ships `tau_relax: float = 15.0` (days) while `atmosphere/physics/convection/config.py:107` ships `tau_relax: float = 7200.0` (seconds). `land/soil_thermal.py:36 C_water=4.18e6` (J/m³/K, volumetric) vs `coupler/lake/config.py:15 c_water=constants.c_pw` (J/kg/K, specific). **Rule (firmed):** any flagged 2026-05-03 unit-suffix item still unfixed at next audit MUST be renamed in same PR as its next downstream use. Block any new code that touches `tau_relax`, `C_water`, or `c_water` without fixing the naming.
- **Driver↔scheme schema field-name drift creates silent aliasing in `model_driver.py:279-280`.** `driver/config.py` ships `hyperdiff_scale` while `ocean/state.py:109,195,267` carries `hyperdiff_coeff`. Same pattern for `topo_smoothing`/`smoothing_passes` and `topo_edge_blend`/`edge_blend_strength`. **Rule:** driver-config field name MUST exactly match the scheme-config field it maps to. Add a unit test `tests/unit/test_driver_scheme_field_alignment.py` that walks `ExperimentConfig` schemas and asserts every flattened field name resolves to an identically-named scheme-config field, or fails with the offending pair.
- **`CAPE_threshold` vs `cape_threshold` capitalization split.** `SBMConfig.CAPE_threshold` (uppercase) vs all other convection schemes using `cape_threshold`. `physics_pipeline.py:1008` constructs `SBMConfig(CAPE_threshold=...)`, `training/aimip_params.py` mixes `sbm_CAPE_threshold` and `tiedtke_cape_threshold`. **Rule:** snake_case for all NamedTuple fields, even when the physical symbol is conventionally capitalized (CAPE, CIN, MSE, TKE). Update SBMConfig + every call site in same PR.
- **`n_layers` triple-overloaded.** `land/soil_grid.py:23` uses it for 8 soil layers; `atmosphere/physics/microphysics/config.py:161` + `gravity_wave_drag/config.py:206` for NN hidden-layer count; `driver/config.py:203` uses `physics_parameterization_layers`. **Rule:** `n_layers` is reserved for atmosphere/ocean vertical-discretization layer count (synonym for `nlev`, see Open naming debt). Soil renames to `n_soil_layers`. ML configs rename to `n_hidden_layers`.
- **One redundant source file remains:** `src/legoesm/core/operators_fv_cubed.py` (63 LoC, single caller, only contains `default_div_damp_coeffs`). **Action:** inline at caller or fold into `operators_fv.py`, drop the file + its `__init__.py` re-export + `supported_matrix.py` entry.
- **Two refactorable script clusters:** `scripts/run_drake_momentum_budget{,_implicit,_divdamp}.py` — 3 scripts differing only by `barotropic_solver` and `barotropic_div_damp` settings. Merge via `--variant {baseline,implicit,divdamp}` flag, saves ~165 LoC. `scripts/run_held_suarez_rrtmgp_allgrids_test.py` is a near-copy of `_4grids.py`. Collapse via `--smoke`/`--quick` flag, saves ~363 LoC.
- **Direct-unit-test debt for 25 high-priority untested LIVE files.** Worst (zero direct test, 36+ script refs / sits in dispatch path): `ocean/experiments/global_overturning.py`, `atmosphere/physics/convection/_triggers.py`, `coupler/surface_energy.py`, `timestepping/tridiagonal.py`, `ocean/dynamics/barotropic_common.py` (CLAUDE.md explicitly says must be reused — needs test enforcing the reuse), `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. **Rule:** any new module added to `__init__.py` re-export, `supported_matrix.py`, or factory dispatch MUST have a direct unit test in the same PR. Block PRs that grow the untested-LIVE count.
- **Empty `src/legoesm/ml/sfno_s2s/` directory removed in this cycle** (contained only `__pycache__`, stale sibling of `ml/s2s/`).

## Common Mistakes to Avoid
Recurring mistakes caught by slopbuster. Check before submitting code:

### NamedTuple field names
- Accessing NamedTuple fields: **verify actual field name** — not what you think it should be. Example: `PhysicsOutput` has `precip`, not `precipitation`. `hasattr` guard silently degrades to fallback instead of catching typo.
- Adding fields to NamedTuple (e.g., `SegmentCarry`): **update every call site** that constructs the NamedTuple. Search with `grep -rn "SegmentCarry(" --include="*.py"` for all constructors. Missing field causes runtime error, but tests in other files may not run until CI catches it.

### Land mask and face masks (latlon C-grid)
- **Never use `state._replace(land_mask=...)` on `LatLonCGridOceanState`** without also updating `u_mask` + `v_mask`. Stale face masks allow mass flux through walls, causing silent mass leaks.
- **Preferred**: pass correct `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction time.
- **Post-construction replacement needed**: use `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomically updates all three masks.
- Runtime check in `_assert_runtime_invariants` (gated by `enable_runtime_checks`) catches inconsistencies.

### Reuse before writing
- **Column integrals**: use `diagnostics.column_integrals.column_water_vapor()` — do not inline `jnp.sum(q * p_s * dsigma) / g`.
- **Loss functions**: import from `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`) — do not reimplement.
- **Optimizer setup**: use `ml/training.create_optimizer()` for warmup + cosine decay + grad clipping — do not inline bare `optax.adam()` without schedule.
- **SFNO model**: import from `ml/sfno.py` — do not create new neural operator architectures in training code.
- **Channel packing**: import from `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`) — do not reimplement state↔tensor conversion.
- **Ocean baroclinic helpers** (#214): for the EOS-pressure iteration, tracer sponge, freshwater virtual-salt flux, or implicit bottom-drag factor in any new ``ocean_pe_*.py`` (or refactor of an existing one), use the helpers in ``src/legoesm/ocean/dynamics/ocean_tendency_common.py`` (``iterate_eos_and_pressure_anomaly``, ``apply_sponge_tracer_relaxation``, ``apply_freshwater_virtual_salt_top``, ``implicit_bottom_drag_factor``). Do not re-inline the 2-pass loop or the sponge cast pattern.
- **Ocean barotropic helpers** (#214): for the cosine/box time filter, BEBT eta blend, or MAXVEL clip in any new ``barotropic_*.py``, use ``src/legoesm/ocean/dynamics/barotropic_common.py`` (``compute_filter_weights``, ``bebt_blend``, ``maxvel_clip``). The structural enforcement test ``tests/ocean/unit/test_no_scheme_duplication.py`` will fail if these are reinlined.

### JIT and compilation
- **Never build closures inside training loops**: `build_segment_fn` creates a new function object each call. If called inside a `for epoch` loop or inside `_loss_fn`, it causes JIT recompilation every iteration. Build once outside the loop; pass changing values as explicit arguments.
- **Helper functions inside `lax.scan` bodies**: Python function definitions inside `_single_step` (the scan body) are recreated every trace. Move helpers (e.g., `_match_dtype`) to module scope.
- **Dead code from iteration**: when refactoring, search for variables that were assigned but never used (e.g., building a segment function then immediately rebuilding inside a nested `_loss_fn`).

### Imports
- Do not import private (`_`-prefixed) functions from other modules. If you need internal functionality, add a public wrapper in the source module.
- Remove unused imports before committing. Check with `grep -n "^from\|^import" <file>` and verify each is used.

### SegmentCarry discipline
- `SegmentCarry` is the canonical hot-loop state. Adding a field is a **cross-cutting change** — update: the NamedTuple definition, `pack_carry`, `unpack_carry` docstring, the per-step Python reference loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, and any direct `SegmentCarry(...)` constructors in validation tests.
- New carry fields used only for diagnostics (e.g., `max_cfl`) should be reset to zero at the start of each segment, not accumulated across segments.

## Physics Constants and Shared Functions
- **All physical constants** (g, R_d, R_v, c_pd, L_v, T_freeze, sigma_sb, etc.) are defined in `src/legoesm/constants.py`. Never redefine these values locally — always `from legoesm import constants` or import the specific name.
- **RRTMGP constants** (`radiation/rrtmgp/constants.py`, `optics/constants.py`) re-export from `legoesm.constants` with SCREAMING_CASE aliases. Do not add new hardcoded values there.
- **NamedTuple config defaults** (e.g. `BulkFormulaConfig.c_pa`, `SeaIceConfig.rho_ice`) should use literal floats matching the central constants. Add a comment referencing the canonical name (e.g. `# = constants.c_pd`).
- **Ocean-specific constants** (`rho_0`, `c_sw`, `T_freeze_ocean`, `scale_depth`) live in `ocean/eos.py`. The ocean freezing point (271.35 K for seawater) is intentionally different from freshwater `T_freeze = 273.15 K` in `constants.py`.
- **Ocean physics helpers** (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`) live in `ocean/eos.py`. Do not duplicate EOS + hydrostatic pressure calls in ocean integration bridges.
- **Saturation thermodynamics**: use `legoesm.thermo` for all saturation computations:
  - `saturation_vapor_pressure(T)` → e_sat [Pa]
  - `saturation_mixing_ratio(T, p)` → q_sat [kg/kg]
  - `saturation_mixing_ratio_ice(T, p)` → q_sat_ice [kg/kg]
  - Do not inline Tetens/Magnus/Clausius-Clapeyron formulas anywhere.
- **Atmosphere column helpers** (hydrostatic heights, density, virtual temperature): use `atmosphere.physics._shared`. Do not duplicate in integration bridges.
- **Function defaults** for `g` should use `constants.g`, not the literal `9.80616`. Same for other constants used as default parameter values.
- **When adding a new parameterization** (atmosphere, ocean, or land): import thermodynamic helpers from `thermo.py`, constants from `constants.py`, ocean constants from `eos.py`. Do not copy-paste from neighboring schemes.

## Existing Claude Assets
- Specialized agents already exist under `.claude/agents/` for dycore expertise, validation, differentiability, physics, land/ice, scalability.
- Use those specialized agents when task deep in one of those domains rather than handling everything as generic coding work.

## Response Style
- Be precise + concrete.
- State assumptions explicitly.
- Changing numerics or algorithms: explain expected effect on stability, accuracy, conservation, differentiability.
- No present guesses as facts.
- **No read/load image files into context** (via Read tool) unless user explicitly asks to see them. Instead, report file path so user can open themselves. Loading images bloats context rapidly.

# iterate-with-codex agent
1. Implement requested change
2. Run /codex:adversarial-review --wait
3. Parse review output
4. Fix all flagged issues
5. Run /codex:review --wait again
6. Issues remain: go to step 4
7. Stop when review clean or after 30 iterations