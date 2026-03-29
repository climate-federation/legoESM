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

## Operating Mode
- For any nontrivial task, start with a short plan before editing.
- Read nearby implementation and tests before proposing or making changes.
- If a request is ambiguous and could affect numerics, physics, APIs, or scientific conclusions, ask a clarifying question before editing.
- Prefer minimal, local diffs. Do not refactor unrelated code during targeted bug fixes.
- Reuse existing shared functions, operators, diagnostics, initial-condition builders, and init/load paths whenever possible.
- Before adding new helpers or new initialization logic, search for an existing implementation that can be extended or factored into a shared location.
- Avoid duplicating numerics across dycores, physics packages, grids, or test setups when a common implementation is feasible.
- Do not trade correctness for speed by skipping validation or making speculative edits.

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
- Atmosphere test matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py`
- Ocean test matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py`
- AMIP production: `.venv/bin/python scripts/run_amip.py`
- Dycore progression suite: `.venv/bin/python tests/validation/run_dycore_progression_suite.py`

## How To Think About Bugs
- For instability: check CFL, boundary treatment, metric terms, halo exchange, pressure-gradient formulation, diffusion, and dtype first.
- For conservation drift: inspect flux form, area/volume weights, reductions, and state updates before adding fixers.
- For differentiability failures: inspect control flow, shape changes, side effects, checkpointing, and nondifferentiable branches.
- For performance regressions: look for retracing, host callbacks, excessive scatters, poor sharding, and accidental Python loops.
- For cross-backend discrepancies: inspect dtype assumptions, x64 requirements, unsupported kernels, and communication semantics.

## Code Hygiene Rules
- Every new `.py` source file must have at least one test that imports and exercises it. Do not add files to `__init__.py` lazy imports or `supported_matrix.py` without a corresponding test.
- New config dispatch branches (new Literal values in config NamedTuples + factory cases in `integration.py`) must have a test exercising that branch.
- When removing a source module, also remove: its `__init__.py` re-export, its `supported_matrix.py` entry, its dispatch entry, its test file, and any stale `__pycache__` files.
- Do not add deprecated backward-compatibility wrappers. If an API changes, update call sites directly.
- Grid-specific variants are legitimate when they have genuinely different numerics. Copy-paste with only indexing changes is forbidden — factor shared logic into a common function.
- Run the slopbuster agent (`/slopbuster audit all` or `/slopbuster review`) periodically, especially before releases.

## Common Mistakes to Avoid
These are recurring mistakes caught by slopbuster. Check for them before submitting code:

### NamedTuple field names
- When accessing NamedTuple fields, **verify the actual field name** — not what you think it should be. Example: `PhysicsOutput` has `precip`, not `precipitation`. A `hasattr` guard silently degrades to a fallback instead of catching the typo.
- When adding fields to a NamedTuple (e.g., `SegmentCarry`), **update every call site** that constructs the NamedTuple. Search with `grep -rn "SegmentCarry(" --include="*.py"` for all constructors. Missing a field causes a runtime error, but tests in other files may not run until CI catches it.

### Reuse before writing
- **Column integrals**: use `diagnostics.column_integrals.column_water_vapor()` — do not inline `jnp.sum(q * p_s * dsigma) / g`.
- **Loss functions**: import from `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`) — do not reimplement.
- **Optimizer setup**: use `ml/training.create_optimizer()` for warmup + cosine decay + grad clipping — do not inline bare `optax.adam()` without schedule.
- **SFNO model**: import from `ml/sfno.py` — do not create new neural operator architectures in training code.
- **Channel packing**: import from `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`) — do not reimplement state↔tensor conversion.

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
