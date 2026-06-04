# legoESM Claude Memory

## Role
- Act like a senior JAX engineer and Earth system model developer.
- Default to careful, skeptical, verification-first work rather than fast iteration.
- Optimize for scientific correctness, physical consistency, differentiability, and maintainability.
- If model selection is available, prefer `opusplan` or `opus` with higher effort for nontrivial dycore, physics, parallel, or debugging work.
- Keep fast mode off for scientific implementation, numerical debugging, and architecture changes unless the user explicitly prioritizes latency.

## Repo Facts
- This repository is a differentiable Earth system model in JAX spanning atmosphere, ocean, land, sea ice, coupler, DA, and ML components.
- End-to-end `jax.grad` compatibility is a design goal. Do not break autodiff, JIT, or pytree semantics for convenience.
- Conservation matters: mass is a hard constraint, and energy/momentum consistency should be preserved whenever the scheme permits.
- The canonical parallel entry point is `ParallelRuntime.create()`.
- The codebase supports cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, and icosahedral pathways.

## Training Infrastructure (`src/legoesm/training/`)
- **Three training modes**: physics parameter tuning, neural GCM, SFNO coupled to dycore.
- All modes use `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad` for AD compatibility.
- `SegmentForcing` is an explicit argument to `run_segment`, not closure-captured — prevents recompilation when forcing changes.
- `TrainablePhysicsParams` wraps 8 physics parameters as an Equinox module with sigmoid constraints.
- ERA5 data: `era5_to_state.py` handles lat-lon → model grid conversion with local Zarr cache.
- Losses: `training/losses.py` imports from `ml/loss.py` — never duplicate loss functions.
- **MPI AD compatibility**: `global_sum_mpi` (allreduce SUM) has full VJP support; MPI halo exchange uses `_sendrecv_vjp` custom_vjp wrapper. Conservation fixers (`fix_mass`, `zero_mean_tendency`) flow gradients correctly through global reductions. `global_max_mpi` / `global_min_mpi` are NOT differentiable — keep them out of loss functions.

## Operating Mode
- For any nontrivial task, start with a short plan before editing.
- Read nearby implementation and tests before proposing or making changes.
- If a request is ambiguous and could affect numerics, physics, APIs, or scientific conclusions, ask a clarifying question before editing.
- Prefer minimal, local diffs. Do not refactor unrelated code during targeted bug fixes.
- **Mandatory pre-implementation search**: BEFORE writing any new function, helper, class, operator, diagnostic, init/load path, loss, or numerical routine, you MUST first search the codebase for an existing implementation. Use Grep/Glob/Explore to look across `src/legoesm/` for: similar function names, similar docstrings, similar formulas, and existing modules in the relevant subpackage (e.g., `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`). State explicitly in your response what you searched for and what you found before adding new code. If something similar exists, extend or factor it — do not duplicate.
- **Always use existing shared utilities by default — never re-derive.** This is the default for *every* task — production source, scripts, validation harnesses, plotting code, tests, notebooks, even one-off probes. Specifically:
  - **Constants**: `from legoesm import constants` — `constants.T_freeze`, `constants.R_d`, `constants.c_pd`, `constants.L_v`, `constants.R_v`, `constants.epsilon`, `constants.g`, `constants.p_ref`, `constants.kappa`, `constants.sigma_sb`, `constants.T_freeze_ocean`, etc.  Never write literals like `273.15`, `287.0`, `1004.64`, `2.501e6`, `461.51`, `0.622`, `9.80616` in any file (production, scripts, tests, plotters).  This is audit-enforced for `src/legoesm/`; the same rule extends to `scripts/` and `tests/`.
  - **Saturation thermodynamics**: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice` — never re-implement Tetens / Magnus / Clausius–Clapeyron in any file (plotters included).  A plotter that re-derived `e_sat = 611.2*exp(17.67*Tc/(Tc+243.5))` produced a *different* q_sat than the model's saturation adjustment and falsely flagged supersaturation in a CI check; the rule exists because formula disagreement masquerades as a physics bug.
  - **Column integrals**: use `legoesm.diagnostics.column_integrals` (e.g. `column_water_vapor`).
  - **Loss functions**: import from `ml/loss.py`.
  - **Optimizer setup**: `ml/training.create_optimizer()`.
  - **Atmosphere column helpers** (hydrostatic heights, density, virtual temperature): `atmosphere.physics._shared`.
  - **Ocean EOS / pressure**: `ocean.eos`.
  - **Plotting** of model output: when computing diagnostic quantities (q_sat, RH, density, virtual T, MSE, etc.) inside a plotter, *import* the model's helper instead of re-deriving.  Plotters are NOT exempt from the "no re-derivation" rule.
  When in doubt, search `src/legoesm/` for the function before writing one.
- Avoid duplicating numerics across dycores, physics packages, grids, or test setups when a common implementation is feasible. Copy-paste with only indexing or naming changes is forbidden.
- Do not trade correctness for speed by skipping validation or making speculative edits.
- **No laziness on hard problems or large code production**: when a task is challenging (numerical bug hunts, dycore ports, multi-file refactors, new parameterizations) or requires substantial code (>100 LOC, multi-component changes, full operator chains, large test matrices), you MUST do the full work. Do not stub functions with `pass` or `raise NotImplementedError`. Do not write a partial implementation and call it done. Do not skip the harder corner cases (edge cells, boundary halos, corner stencils, non-duogrid branches, MPI/sharded paths, AD/VJP support) and quietly leave them for later. Do not abbreviate test coverage to a single happy path. If the task is genuinely too large for one pass, say so explicitly, list every piece that remains, and quantify the residual risk — never imply completion you have not delivered.

## JAX Engineering Rules
- Keep functions pure and pytree-friendly.
- Prefer `jax.lax.scan` for time integration and structured loops.
- Prefer `jax.vmap` or batched array expressions over Python loops on array dimensions.
- Use `jnp.where`, `jax.lax.cond`, `jax.lax.fori_loop`, or `scan` instead of Python control flow on traced values.
- **But**: for on/off feature gating (e.g., `fix_mass`, `fix_moisture`), use Python `if` on a static bool captured in the closure — NOT `jnp.where`, which traces both branches and wastes compute. `jnp.where` is for data-dependent selection on traced values only.
- Preserve stable shapes and avoid unnecessary retracing.
- Be explicit about dtype behavior. Spectral solvers expect x64 and complex128; finite-volume pathways may intentionally run in float32.
- Avoid host/device thrash, unnecessary materialization, or ad hoc NumPy fallbacks inside traced code.
- Do not introduce hidden non-JAX side effects that break JIT, grad, checkpointing, or sharding.
- **Buffer donation and `jax.grad`**: `@jax.jit(donate_argnums=...)` frees input buffers after the call. This conflicts with reverse-mode AD, which needs inputs for the backward pass. When a JIT-compiled function will be called inside `jax.grad` or `eqx.filter_value_and_grad`, provide a non-donating variant (e.g., `.raw` attribute) and use that for training. See `build_segment_fn` for the pattern.
- **Closures vs explicit args for JIT reuse**: values captured in a Python closure become compile-time constants. If a value changes every iteration (e.g., SST, solar forcing), pass it as an explicit traced argument — not a closure capture — so the compiled kernel is reused. See `SegmentForcing` for the pattern.

## Earth System Modeling Rules
- Treat conservation, metric consistency, staggered-grid consistency, and halo correctness as first-class requirements.
- Never "fix" numerical problems with silent clipping, damping, or coercion unless scientifically justified and validated.
- Preserve units, sign conventions, monotonicity/positivity assumptions, and hydrostatic/nonhydrostatic consistency.
- For cubed-sphere and other curvilinear grids, assume edge and metric errors are likely root causes until ruled out.
- For physics coupling, maintain column closure and physically consistent flux signs across atmosphere, land, ocean, ice, and coupler interfaces.
- For DA and differentiable workflows, preserve smoothness where intended and avoid gratuitous nondifferentiable logic.

## Parallel and HPC Rules
- Preserve correctness under serial, multi-device, MPI, and hybrid execution.
- For sharded or distributed code, reason explicitly about halo exchange, reduction semantics, partition specs, and global invariants.
- Validate single-rank behavior before assuming a distributed bug is fixed.
- Then verify rank/device equivalence on the smallest meaningful distributed case.
- Do not assume Metal, GPU, CPU, spectral, and MPI backends have identical dtype or kernel constraints.
- On Apple Silicon, keep spectral work on CPU.
- **MPI distributed path**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local stepping → `gather_to_global()` for I/O only. Both ModelDriver and the benchmark script use this path. Never create a full global state per rank — always scatter.
- **Native 4D halo exchange**: `pad_halo_4d()` and `pad_halo_vector_4d()` exchange all vertical levels in one MPI message. All 3D operators in `operators_3d.py` use the 4D path. Do not revert to `vmap(pad_halo)` which issues `nlev` separate messages.
- **MPI halo AD safety**: All `sendrecv` calls go through `_sendrecv_vjp` (`@jax.custom_vjp` wrapper in `halo_exchange.py`) that swaps source/dest in the backward pass. This enables `jax.grad` through MPI halo exchange. Only `allreduce(SUM)` is AD-safe among MPI reductions; `MAX`, `MIN`, `allgather`, and `bcast` are not differentiable — use only in diagnostics.
- **Device mesh under MPI**: Pass per-rank device count to `create_device_mesh()`, not the total across all ranks. The function warns when clamping.

## Validation Rules
- Always run the narrowest relevant test after edits.
- For numerical changes, prefer analytical or benchmark-style validation rather than relying only on unit tests.
- Use `JAX_ENABLE_X64=1` for most scientific validation unless the task is specifically about float32 or Metal portability.
- If touching dycore operators, consider Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, or equivalent ocean benchmarks as appropriate.
- If touching conservation, reductions, or coupler logic, check mass and energy diagnostics explicitly.
- If touching parallel code, verify unsharded vs sharded or single-rank vs MPI agreement.
- If full validation is too expensive, say exactly what was run, what was not run, and what residual risk remains.
- **CRITICAL — Visual verification for spatial/grid artifacts**: Passing unit tests and error norms is NECESSARY but NOT SUFFICIENT when modifying cubed-sphere operators, halo exchange, diffusion coefficients, or grid metrics. Edge artifacts, cube imprint, and grid-scale noise are ONLY reliably detected by visual inspection of field snapshots (especially v-wind in Williamson 2, wind_speed in Williamson 5). Always run the atmosphere test matrix quick mode (`--only sw --grid cubed_sphere --quick`) and inspect the generated snapshot PNGs before claiming a fix works. Compare against a known-good baseline image. Error norms can improve while visual artifacts get worse (e.g., if artifacts shift location or change character). Never claim "tests pass, edge artifacts fixed" based on pytest results alone.
- **Diffusion coefficient sensitivity**: Divergence damping and hyperdiffusion coefficients AMPLIFY halo-exchange gradient errors at cubed-sphere face boundaries. Increasing these coefficients (even modestly) can worsen edge artifacts. Always check visual impact on Williamson 2 v-wind when changing `_hyperdiff_cube`, `_div_damp_cube`, or any diffusion parameter.

## Project-Specific Commands
- Install: `pip install -e ".[dev]"`
- General tests: `.venv/bin/python -m pytest tests/`
- Targeted scientific tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atmosphere test matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean test matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP production: `.venv/bin/python scripts/run_amip.py`
- Dycore progression suite: `.venv/bin/python tests/validation/run_dycore_progression_suite.py`
- GPU/MPI scaling benchmark: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (see `docs/REAL_HARDWARE_SCALING.md`)
- MPI distributed tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI differentiability tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## How To Think About Bugs
- For instability: check CFL, boundary treatment, metric terms, halo exchange, pressure-gradient formulation, diffusion, and dtype first.
- For conservation drift: inspect flux form, area/volume weights, reductions, and state updates before adding fixers.
- For differentiability failures: inspect control flow, shape changes, side effects, checkpointing, and nondifferentiable branches.
- For performance regressions: look for retracing, host callbacks, excessive scatters, poor sharding, and accidental Python loops.
- For cross-backend discrepancies: inspect dtype assumptions, x64 requirements, unsupported kernels, and communication semantics.

## Code Hygiene Rules
- Every new `.py` source file must have at least one test that imports and exercises it. Do not add files to `__init__.py` lazy imports or `supported_matrix.py` without a corresponding test.
- New config dispatch branches (new Literal values in config NamedTuples + factory cases in `integration.py`) must have a test exercising that branch.
- Every dispatch branch must have at least one test that selects it via the public config (not just a unit test of the leaf module).
- When removing a source module, also remove: its `__init__.py` re-export, its `supported_matrix.py` entry, its dispatch entry, its test file, and any stale `__pycache__` files.
- Do not add deprecated backward-compatibility wrappers. If an API changes, update call sites directly.
- Do not add thin dispatch-only wrappers (e.g., `X_utils.py` that just re-exports a function from `X.py`). Inline the call at each site or factor the helper into the canonical module. (Modules with real branching/dispatch logic across multiple callers — like `land/stomata_utils.py` — are legitimate and not "thin wrappers".)
- Grid-specific variants are legitimate when they have genuinely different numerics. Copy-paste with only indexing changes is forbidden — factor shared logic into a common function.
- **Never commit anything from `docs/references/`.** This directory is a local-only stash for research PDFs (papers, technical reports, multi-MB figures) consulted while planning. It must not enter the git history under any circumstances — not the PDFs themselves, not extracts, not derivative summary files placed in that folder. Read freely; cite by filename or DOI in commit messages and design docs; place any extracted notes elsewhere (`docs/ocean_experiments/`, etc.). When staging files, use explicit paths — never `git add .` or `git add -A` in this repo, since `docs/references/` is not currently enforced by `.gitignore`.
- Run the slopbuster agent (`/slopbuster audit all` or `/slopbuster review`) periodically, especially before releases.

## Constant and Parameter Discipline (audit-enforced)
- **All physical constants live in `src/legoesm/constants.py`. PERIOD.** Any new physical constant — temperature, density, specific heat, latent heat, thermal conductivity, viscosity, EOS coefficient, Kell density-curvature, Schmidt number, Earth radius, etc. — MUST be added to `legoesm.constants` BEFORE being used anywhere else.  Do NOT introduce new physical constants directly in `config.py` files, function bodies, test fixtures, plotting scripts, or notebook cells — even with a `# = constants.X` annotation comment. The rule is: if you find yourself wanting to write `T_max_dens = 277.133  # K` inside a physics module, STOP, add `T_freshwater_max_density = 277.133` to `constants.py`, then `from legoesm import constants` and reference `constants.T_freshwater_max_density`. This applies whether you are writing production code in `src/legoesm/`, a scheme `config.py`, a test, a plotter, or a one-off probe.
- **`getattr(..., "radius", <literal>)` fallbacks count as hardcoded constants.** A pattern like `R = getattr(grid, "radius", 6.371e6)` ships a hardcoded `6.371e6` default that drifts from `constants.R_earth = 6.371229e6`.  Always use `getattr(grid, "radius", constants.R_earth)` (or import the specific constant) so the fallback path stays in sync with the canonical value.  Same rule applies to `setattr` defaults, `dict.get` defaults, and `kwargs.get(..., <literal>)`.
- **No hardcoded physical constants in function signatures or function bodies in production code (`src/legoesm/`).** Function defaults like `def f(g=9.80616, ...)` are forbidden — use `from legoesm import constants` and `g: float = constants.g`, or read from a config NamedTuple. This includes ocean experiment scaffolding: module-level patterns like `_G_EARTH = 9.80616` must be replaced with `constants.g`. **NamedTuple field defaults that duplicate a physical constant SHOULD reference the constants module** (`rho_water: float = constants.rho_water` preferred over `rho_water: float = 1000.0  # = constants.rho_water`).  Existing NamedTuples with annotated literal defaults are grandfathered, but new fields must use the `constants.X` reference.  Truly tunable scheme parameters (sigmoid sharpness, relaxation timescales, drag coefficients) stay in their config NamedTuple — those are scheme tunings, not physical constants.
- **No hardcoded tunable parameters in physics function bodies.** Sigmoid sharpness, relaxation timescales, Louis coefficients, KPP epsilon, surface emissivity, drag coefficients, etc. must live in the scheme's config NamedTuple with a documented default. Numerical safety floors (`eps=1e-30` for division) and pure mathematical constants (`0.5` in midpoint averaging, `2.0` in squared norms) are exempt. Lookup tables that legitimately vary the constant by category (e.g., PFT-dependent `L_v` rounding in `surface_params.py`) are exempt — but document why.
- **No `273.15` for Celsius↔Kelvin conversions in production code.** Import `constants.T_freeze`. The ocean freezing point (271.35 K) in `ocean/eos.py` is the only intentional exception — it stays a literal with a comment.

## Naming Discipline
- Surface temperature is `T_sfc` everywhere — atmosphere, land, coupler, diagnostics, ocean. Do not introduce new `T_surface` or `Ts` fields. (Existing split is tracked tech debt; see "Open naming debt" below.)
- Driver/config schema field names must match the runtime field they map to. When you add a new tunable, use the same name in `driver/config.py`, the scheme's config NamedTuple, and any YAML schema (e.g., consistent `hyperdiff_coeff`, not `hyperdiff_scale` in one and `hyperdiff_coeff` in another).
- **Same name, different units, is a bug magnet — and now caught by audit.** If a parameter name has a physical-unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`, etc.), use it everywhere that quantity appears. Slopbuster audit 2026-05-13 renamed the ocean-init Celsius initial-condition arg from `T_surface: float = 20.0 (°C)` to `T_water_init_C` to disambiguate against the atmospheric/coupler/land `T_sfc` (or surviving `T_surface`) Kelvin tensor field. Future Celsius-vs-Kelvin scalar args MUST carry the `_C` / `_K` suffix.

### Open naming debt
- **`T_sfc` (184 sites) vs `T_surface` (~15 sites in coupler/land Kelvin tensors)** — the surface-skin-temperature field. Coupler `coupling_fields.SurfaceToAtm.T_surface` and `TileResponse.T_surface` (Kelvin pytree leaf) plus `land/multilayer_land.py`, `land/snow_budget.py`, `land/stomata_utils.py`, `land/slab_land.py`, `ice/sea_ice.py`, `coupler/lake/two_layer_lake.py`, and `coupler/accumulator.py` still carry `T_surface` / `sum_T_surface` field names. Unify to `T_sfc` in a dedicated cleanup PR — not bundled into unrelated work.
- **`nlev` (3566 sites) vs `n_levels` (143) vs `nz` (5)** — vertical-level count uses three names. `nlev` dominates. Unify in a dedicated cleanup PR.

## Untested-but-live debt (audit-enforced)
- New physics schemes (ocean vertical mixing, atmosphere turbulence, convection, microphysics) must have at least one direct unit test that imports the leaf module and exercises its tendencies — not just an integration test that hits it via the factory. The 2026-04-28 audit found 12 high-risk leaf modules in `ocean/physics/` and `atmosphere/physics/turbulence/` that are reachable only through `integration.py` with no direct test. Do not extend this pattern.

## Audit lessons — 2026-05-03 cycle
- **Eager top-level imports across packages cause silent CI fragility.** A single `from legoesm.runtime.backend import ...` at the top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking every isolated `pytest tests/ocean/unit/test_X.py` invocation while bulk collection happened to paper over it. **Rule:** when a low-level package (`core/`) needs symbols from a higher-level orchestration package (`runtime/`, `driver/`), defer the import inside the function/method that needs it. Do not add eager top-level cross-package imports that re-enter through `__init__.py`.
- **`scheme="..."` factories must `raise ValueError` for unknown literals.** `cloud_fraction.compute_cloud_properties` previously had `if scheme == "xu_randall": ... else: sundqvist`, so any unknown literal silently ran as sundqvist. Every dispatch site in `integration.py` already does `raise ValueError(f"Unknown ... scheme: {config.scheme!r}")`; cloud_fraction now does too. **Rule:** every public-config dispatch must validate the scheme literal explicitly and raise on unknown values. Silent fallbacks mask test typos and dead branches. Add the matching validation in `ExperimentConfig.validate_strict` whenever a new scheme literal is introduced.
- **Same field name, different units, is a bug magnet.** Audit found `tau_relax` meaning *seconds* in `convection/config.py` and *days* in `ocean/experiments/phillips_two_layer.py`; `c_water` (J/kg/K) in `lake/config.py` vs `C_water` (J/m³/K) in `land/soil_thermal.py`; `T_ice` used as a freezing-point parameter in `driver/config.py`/`forcing/amip.py` when it is actually `T_freeze_ocean = 271.35 K`. **Rule:** when two config fields share a base name (`tau_*`, `T_*`, `c_*`, `C_*`), they must use consistent unit suffixes (`_s` / `_days`, `_K` / `_C`, `_specific` / `_volumetric`) OR clearly distinct names. Do not let two NamedTuples carry the same identifier with different physical meaning.
- **Hardcoded function defaults for physical constants are forbidden in `src/legoesm/`.** Audit found ~25 sites with `def f(g=9.80616, ...)` or `def f(p_s_init: float = 1.0e5, ...)` style defaults that drift from `constants.g`/`constants.p_ref`. **Rule:** function signatures in `src/legoesm/` must use either `g: float = constants.g` (preferred) or read the value from a config NamedTuple — never a bare literal. NamedTuple field defaults may use literal floats with a `# = constants.X` comment per the existing exception.
- **Tests and scripts must follow the constant-hygiene rule too.** The 2026-05-03 audit found 80+ test sites and 30+ `scripts/diag_*.py` sites still hardcoding `g = 9.80616` and `omega = 7.292e-5`. **Rule:** new tests, plotters, and diagnostic scripts must `from legoesm import constants` and use named constants. Use `constants.g`, never `9.80616`.
- **Test-only modules must be acknowledged as test-only.** Files (`core/fv3_d_sw5_corner_divergence.py`, `ocean/dynamics/ocean_pe_fc.py`, `core/_future/vertical_remap.py`, `ocean/diagnostics.py`) reachable *only* from their own unit tests — not from any production driver, factory, or `supported_matrix.py` entry — are tech debt. **Rule:** if a `src/legoesm/` module is not wired into any factory dispatch, `__init__.py` re-export, or production driver, either (a) wire it in within the same PR that adds it, (b) move it to a `_future/` subpackage with a docstring stating it is staged-not-integrated and mark its tests `@pytest.mark.xfail` or `@pytest.mark.skip(reason="not yet wired")`, or (c) delete it. No "alive only because a test imports it" modules.

## Audit lessons — 2026-05-13 cycle
- **`getattr(grid, "radius", 6.371e6)` fallbacks count as hardcoded constants.** Slopbuster Pass 9 (2026-05-13) found three sites (`ocean/dynamics/latlon_cgrid_operators.py`, `ocean/diagnostics_streamfunction.py` ×2) shipping a hardcoded `6.371e6` Earth-radius fallback that drifts from `constants.R_earth = 6.371229e6`. **Rule:** fallback values in `getattr` / `setattr` / `dict.get` / `kwargs.get` MUST be `constants.X`, never a literal. Same rule applies to NamedTuple field defaults.
- **Sigmoid sharpness / transition widths in hot loops must be configurable.** Slopbuster Pass 10 (2026-05-13) found `jax.nn.sigmoid((p - p_lcl)/100.0)` (LCL transition, `atmosphere/physics/thermodynamics.py`) and `jax.nn.sigmoid(delta_rho * 1e4)` (plume active mask, `ocean/physics/convection/plume.py`) hardcoded inside scan bodies. Both have been lifted: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)` and `PlumeConfig.active_sigmoid_sharpness=1e4`. **Rule:** any sigmoid sharpness, transition width, or smoothing scale that lives inside a `scan_step` / `cond` body MUST be a function kwarg with a documented default, or a field on the scheme's `*Config` NamedTuple. Hardcoded magic numbers in JAX hot loops are forbidden.
- **`ml/` subtree is staged-not-integrated experimental scaffolding.** Slopbuster Pass 7 (2026-05-13) found 36 modules in `src/legoesm/ml/` have zero imports from non-`ml/` source — the subtree is fully self-contained.  The publicly re-exported names (`SFNO`, `SFNOBlock`, `SpectralConv`, channel-packing helpers, conservation correctors) ARE covered by direct tests and safe to import from production callers; the rest of `ml/` (especially `ml/s2s/`, `ml/physics/`, `ml/training.py`) is research scaffolding. **Rule:** do not extend the untested-leaf footprint of `ml/`. New work either (a) lands with at least one direct unit test, or (b) goes to a `_future/` sibling per the staged-not-integrated convention. See `src/legoesm/ml/__init__.py` docstring for the full audit status.
- **Direct-unit-test debt was paid down for top hot-loop leaves.** New tests landed for `timestepping/{dispatch,integration,pytree_ops,split_explicit}.py` (`tests/unit/test_timestepping_leaves.py`) and `diagnostics/column_integrals.py` (`tests/unit/test_diagnostics_column_integrals.py`). **Rule:** if you touch any of these modules, extend the corresponding direct-unit-test file in the same PR; do not rely on indirect coverage through dycore drivers.

## Common Mistakes to Avoid
These are recurring mistakes caught by slopbuster. Check for them before submitting code:

### NamedTuple field names
- When accessing NamedTuple fields, **verify the actual field name** — not what you think it should be. Example: `PhysicsOutput` has `precip`, not `precipitation`. A `hasattr` guard silently degrades to a fallback instead of catching the typo.
- When adding fields to a NamedTuple (e.g., `SegmentCarry`), **update every call site** that constructs the NamedTuple. Search with `grep -rn "SegmentCarry(" --include="*.py"` for all constructors. Missing a field causes a runtime error, but tests in other files may not run until CI catches it.

### Land mask and face masks (latlon C-grid)
- **Never use `state._replace(land_mask=...)` on `LatLonCGridOceanState`** without also updating `u_mask` and `v_mask`. Stale face masks allow mass flux through walls, causing silent mass leaks.
- **Preferred**: pass the correct `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction time.
- **If post-construction replacement is needed**: use `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — it atomically updates all three masks.
- The runtime check in `_assert_runtime_invariants` (gated by `enable_runtime_checks`) will catch inconsistencies.

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
- Specialized agents already exist under `.claude/agents/` for dycore expertise, validation, differentiability, physics, land/ice, and scalability.
- Use those specialized agents when a task is deep in one of those domains rather than handling everything as generic coding work.

## Response Style
- Be precise and concrete.
- State assumptions explicitly.
- When changing numerics or algorithms, explain the expected effect on stability, accuracy, conservation, or differentiability.
- Do not present guesses as facts.
- **Do not read/load image files into context** (via the Read tool) unless the user explicitly asks to see them. Instead, report the file path so the user can open it themselves. Loading images bloats context rapidly.

# iterate-with-codex agent
1. Implement the requested change
2. Run /codex:adversarial-review --wait
3. Parse the review output
4. Fix all flagged issues
5. Run /codex:review --wait again
6. If issues remain, go to step 4
7. Stop when review is clean or after 30 iterations
