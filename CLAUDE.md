# legoESM Claude Memory

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` (allreduce SUM) full VJP. MPI halo: `_sendrecv_vjp` custom_vjp. `fix_mass`/`zero_mean_tendency` flow grads via global reductions. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.

## Operating Mode
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run the **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict. Exempt: trivial/mechanical edits (typo, comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe; `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `.venv/bin/python tests/validation/run_dycore_progression_suite.py`
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in the correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick the bucket by what the script DOES. New bucket needs a real category, not a dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` is gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). Never add source at `src/`/repo root. New subpackage → update `tests/test_federation_plan.py` same PR.
- **Tests mirror the package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catches stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/` = 0.
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean_experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` masks typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. STILL silent (follow-up): `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. GAP (2026-05-29 audit): `convection`/`turbulence`/`gravity_wave_drag` have NO validate_strict membership check — typos pass early validation, fail only at JIT inside `integration.py`. Add them.

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrades. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` creates new fn per call → inside `for epoch`/`_loss_fn` recompiles each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style
Precise+concrete. Explicit assumptions. Numerics change → explain effect on stability, accuracy, conservation, differentiability. No guesses as facts. **No Read images** unless user asks; report path.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
