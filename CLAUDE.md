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
- Preserve stable shapes and avoid unnecessary retracing.
- Be explicit about dtype behavior. Spectral solvers expect x64 and complex128; finite-volume pathways may intentionally run in float32.
- Avoid host/device thrash, unnecessary materialization, or ad hoc NumPy fallbacks inside traced code.
- Do not introduce hidden non-JAX side effects that break JIT, grad, checkpointing, or sharding.

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

## Project-Specific Commands
- Install: `pip install -e ".[dev]"`
- General tests: `.venv/bin/python -m pytest tests/`
- Targeted scientific tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Parallel validation entry point: `.venv/bin/python scripts/run_parallel_validation.py`
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

## Existing Claude Assets
- Specialized agents already exist under `.claude/agents/` for dycore expertise, validation, differentiability, physics, land/ice, and scalability.
- Use those specialized agents when a task is deep in one of those domains rather than handling everything as generic coding work.

## Response Style
- Be precise and concrete.
- State assumptions explicitly.
- When changing numerics or algorithms, explain the expected effect on stability, accuracy, conservation, or differentiability.
- Do not present guesses as facts.
