<p align="center">
  <img src="docs/assets/legoESM.png" alt="legoESM" width="400">
</p>

# legoESM v0.1

**A Differentiable Earth System Model in JAX**

legoESM is a next-generation, fully differentiable Earth System Model spanning
weather-to-climate timescales. Built from scratch in JAX, it couples
atmosphere, ocean, land, sea ice, lakes, and a tile-based surface coupler
through a unified interface, and enables end-to-end gradient computation
through the coupled model — for variational data assimilation, parameter
estimation, sensitivity analysis, and hybrid AI–physics modeling.

## Features

- **Differentiable-first**: end-to-end `jax.grad` (and `eqx.filter_value_and_grad`) through the full coupled model, including MPI halo exchange and global reductions
- **Conservation as a hard constraint**: mass conserved by construction; energy and momentum tracked diagnostically with explicit budget closure
- **Modular & swappable** (the "lego" in legoESM): standard tensor-in / tendency-out interfaces let you swap dynamical cores, physics schemes, grids, vertical coordinates, time integrators, and complexity levels independently
- **Multi-grid**: cubed-sphere (C-D grid, FV3-faithful PPM), lat-lon (C-grid), Gaussian/spectral, Voronoi/MPAS icosahedral
- **Hardware-portable**: CPU, multi-GPU, TPU, Apple Silicon (Metal/CPU hybrid), and multi-node MPI / hybrid SPMD execution
- **AI-ready**: SFNO-based neural cores and ML parameterizations co-exist with classical physics under the same interface; ERA5 initialization built-in

## Repository Structure

legoESM is a **uv workspace of independently-installable packages** that all
ship into one shared `legoesm` PEP-420 namespace — so `import legoesm.<subpkg>`
is unchanged, but each Earth-system component can be `pip install`ed and run on
its own (a single `legoesm` meta-package pulls them all). See
[FEDERATION.md](FEDERATION.md) for the dependency DAG and the
import-boundary contracts (`lint-imports`).

### Packages (`packages/<member>/legoesm/<subpkg>/`)

| Member (dist name) | Subpackages | Role | Primary entry points |
|---|---|---|---|
| **legoesm-core** | `core`, `grids`, `runtime`, `parallel`, `io`, `timestepping`, `components` (+ `constants`, `thermo`, `registry`, `surface_albedo`) | The substrate — fields, operators, grids, vertical coords, device/precision runtime, parallel halo/reductions, the grid+complexity capability matrix. Imports nothing above it. | — (library; `legoesm.grids.instantiate`, `ParallelRuntime.create`) |
| **legoesm-atmosphere** | `atmosphere` | Dycores (cubed-sphere C-D, spectral, lat-lon C-grid, MPAS, SFNO), physics packages, SCM. Depends only on core. | `scripts/run/run_amip.py`, `scripts/matrix/run_atmosphere_test_matrix.py`, `scripts/matrix/run_scm_test_matrix.py` |
| **legoesm-ocean** | `ocean` | 3-D ocean dynamics/physics, idealized + realistic experiments, spin-up, fidelity harness. Depends only on core. | `scripts/run/run_omip.py`, `scripts/matrix/run_ocean_test_matrix.py` |
| **legoesm-land** | `land` | Slab + multilayer land, carbon, snow, PFT providers. Depends only on core. | `scripts/run/run_lmip.py` |
| **legoesm-ice** | `ice` | Thermodynamic + EVP/mEVP sea ice, ITD. Depends only on core. | `scripts/matrix/run_sea_ice_test_matrix.py` |
| **legoesm-coupler** | `coupler`, `driver` | Tile-based surface coupler + the drivers (`ModelDriver`, `CoupledESMDriver`, `EarthSystemDriver`) + reproducibility manifest. | `scripts/run/run_coupled.py`, `legoesm run`, `legoesm reproduce` |
| **legoesm-ml** | `ml`, `training`, `da` | SFNO blocks, conservation-aware losses, ERA5 ingestion, differentiable training, variational DA, S2S. | `scripts/run/train_neural_gcm_spectral.py`, `scripts/run/ml_physics_parameterization.py` |
| **legoesm-tools** | `forcing`, `diagnostics`, `experiments`, `visualization` | Forcing loaders, energy/mass diagnostics + conservation gates, the shared **test-matrix framework** (`experiments.matrix`), plotters. | (used by the matrix runners + harness) |

The four components (atmosphere/ocean/land/ice) depend **only** on
`legoesm-core` and never on each other or the coupler — that one-way DAG is what
lets `pip install legoesm-ocean` run standalone. The orchestration cluster
(coupler/ml/tools) sits above them.

### Scripts (`scripts/<bucket>/`)

| Bucket | Contents |
|---|---|
| `run/` | Production experiment drivers (`run_amip`, `run_omip`, `run_rce`, `run_held_suarez*`, `run_coupled`, `run_aimip*`, cross-grid `.sh` wrappers, …) |
| `matrix/` | Complexity-tiered test-matrix runners (`run_{atmosphere,ocean,sea_ice,scm}_test_matrix`, `summarize_matrix_results`, `validate_matrix_report`, `check_conservation_all`) |
| `experiment/` | The template/provenance harness — `init_experiment`, `validate_templates`, `lego_detect_machine`, `fetch_data` |
| `bench/` `plot/` `validate/` `data/` | Benchmarking/scaling · plotting · validators+intercomparison · forcing/data-prep+setup |
| `tmp/` | Throwaway debug/diagnostic scripts (slated for deletion — never wire into CI) |

`config/templates/` holds versioned, tier-annotated experiment templates;
`config/machines/` holds per-host profiles; `config/data_catalog.yaml` maps
external datasets. See [scripts/README.md](scripts/README.md).

### Tests (`tests/`)

Per-component dirs (`atmosphere/`, `ocean/`, `land/`, `sea_ice/`, …) plus
`unit/`, `validation/`, `distributed/`, `tools/`. Complexity-tier markers
`tier0`–`tier3` (research → operational) gate the suite; the curated
FV3/cubed-sphere dycore regressions live in `tests/atmosphere/dycore/regression/`.
See [docs/validation/TESTING.md](docs/validation/TESTING.md) for the shared tier ladder and how a
component plugs into the matrix framework.

## Model Components

### Atmosphere

- **Dynamical cores** (canonical entries in `supported_matrix.py`):
  - **Cubed-sphere C-D grid** (FV3-faithful PPM transport, divergence damping): shallow water, hydrostatic primitive equations, non-hydrostatic compressible Euler
  - **Spectral / Gaussian grid** (semi-implicit, spherical-harmonic transforms, `complex128`): shallow water, hydrostatic, non-hydrostatic
  - **Lat-lon C-grid** (finite volume): shallow water, hydrostatic
  - **MPAS / Voronoi icosahedral**: hydrostatic, non-hydrostatic
  - **SFNO data-driven cores**: shallow water and hydrostatic learned solvers
  - **Tracer transport** modules on cubed-sphere, lat-lon, and Voronoi, with RK3 time stepping for full 3rd-order convergence
- **Single-column model (SCM)** (`legoesm.atmosphere.scm.SingleColumnModel`, `scripts/matrix/run_scm_test_matrix.py`): dycore-free driver that reuses the full physics factory for RCE, GABLS-style boundary-layer cases, parameterization integration tests, and any-scheme × any-integrator swap-matrix sweeps
- **Vertical coordinates**: pure sigma and hybrid sigma–pressure (L20–L60, sinh stretching)
- **Physics packages** (each with a config NamedTuple, factory dispatch in `integration.py`, and direct unit tests):
  - **Radiation**: gray (Frierson-style) and RRTMGP correlated-k (LW + SW) with diurnal cycle, prescribed/transient ozone, aerosols, solar TSI, and cloud–radiation coupling
  - **Convection**: Bechtold, Tiedtke, Kain–Fritsch, Zhang–McFarlane, Emanuel, Kuo, SBM, plus a deep convection adjustment (DCA) — sharing a unified mass-flux core with a stratospheric mass-flux gate
  - **Microphysics**: Kessler, Sundqvist large-scale condensation, Morrison double-moment, Seifert–Beheng, Thompson, plus an ML emulator path
  - **Cloud fraction**: Sundqvist and Xu–Randall
  - **Boundary layer / turbulence**: Louis, Holtslag–Boville, YSU, EDMF, TKE 1.5-order, CLUBB-lite, Smagorinsky, surface-layer Monin–Obukhov, vertical diffusion, with PBL-height diagnosis
  - **Gravity-wave drag**: Lindzen, Hines, McFarlane, Rayleigh, prognostic spectral, ML emulator
  - **Held–Suarez** dry forcing for benchmark runs

### Ocean

- **3D ocean dynamics**: split-explicit baroclinic / barotropic (cubed-sphere C-D, lat-lon C-grid, MPAS Voronoi, FC-Gram) with implicit-barotropic, BEBT-blended free-surface, MAXVEL clip, sponge relaxation, mass-conservation projection, and shared baroclinic helpers (EOS-pressure iteration, virtual-salt freshwater flux, implicit bottom-drag factor)
- **Grids**: lat-lon (regular + Mercator), tripolar (NEMO eORCA1 mesh_mask loader + tensor pole-fold halo), cubed-sphere C-D, MPAS Voronoi (LSQ edge-to-cell + Thuburn kite-area TRiSK); cross-grid metric-consistency test matrix at 57/57 PASS
- **Spectral ocean** (research path) and **SFNO learned ocean**
- **Ocean physics**: KPP / Richardson / constant vertical mixing; Jayne–St-Laurent (2001) abyssal tidal mixing with tracer-mixing integration; harmonic, biharmonic, GM-Redi (cubed-sphere, lat-lon, MPAS variants), Visbeck adaptive-GM, Leith viscosity, and backscatter lateral mixing; convective adjustment (enhanced diffusion + plume); linear / quadratic bottom drag with implicit factor; bulk-formula / restoring / prescribed surface forcing; shortwave penetration (Jerlov)
- **Ice-shelf cavity coupling**: Holland & Jenkins (1999) three-equation basal-melt at the ice base, lat-lon C-grid and MPAS apply paths
- **External forcing**: Dai–Trenberth global river runoff (point→grid projection), OMIP-2 sea-surface salinity restoring + WOA SSS climatology loader, JRA55-do RYF preload (float32-safe for 32 GB GPUs), and an external `tau` / `q_net` / `sw_down` pathway for C-grid PE
- **Biogeochemistry**: abiotic carbon (DIC + ALK with carbonate equilibria and air-sea CO₂ flux) and an NPZD ecosystem
- **Idealized experiments suite** (`ocean/experiments/`): rest state, Eady / Phillips / baroclinic gyres, ACC channel, Drake/Stommel, lock exchange, overflow, baroclinic & barotropic wave, geostrophic adjustment, inertia–gravity wave, global overturning, Silvestri baroclinic jet, Munk, Held–Larichev, NeverWorld2-lite, ISOMIP+
- **Realistic geometry**: NetCDF bathymetry (ETOPO/GEBCO/ERDDAP) with bilinear regridding, Laplacian smoothing, MEO r-cap steepness limiter, polar-cap masking, flood-fill isolated-basin removal, and strait enforcement
- **Centennial spin-up library** (`ocean.spinup`): AMOC@26.5°N tracker, RPE / volume / heat / salt drift diagnostics, declarative `ConvergenceCriteria`, Bryan–Lewis (1984) distorted-physics accelerated protocol, and auto-restart discovery (`scripts/run/run_omip.py`)
- **Peer-comparison fidelity harness** (`ocean/fidelity/`): Veros DINO / Eady adapters, regridder, and `docs/ocean/fidelity/legoesm_vs_veros_v2.md`
- **Simple ocean**: slab mixed-layer and two-layer (cubed-sphere and MPAS variants)

### Land Surface

- **Slab land**: energy balance + bucket hydrology + snow with stomatal conductance (Farquhar + Ball–Berry / Medlyn / Jarvis)
- **Multi-layer land**: Richards equation (mixed-form Picard) with 6 retention curves (Van Genuchten, Clapp–Hornberger, Brooks–Corey, Campbell, PDI, Lu) and Johansen thermal diffusion
- **Carbon cycle**: DALEC-990 6-pool (labile / foliage / root / wood / litter / SOM) with LUE GPP and a seasonal scheme
- **Snow**: accumulation/melt budget with age-dependent albedo and latitude-varying vegetation albedo
- **PFT-weighted parameter providers**: surface roughness, albedo, capacity, conductance

### Cryosphere

- **Sea ice**: thermodynamic slab + free-drift + EVP / mEVP rheology (Hunke & Dukowicz 1997; Bouillon 2013 / Kimmritz 2015) on lat-lon C-grid and MPAS Voronoi with tensor pole-fold halo, multi-category ITD (Lipscomb 2001 linear remap), temperature-dependent albedo, transport
- **Tier 1+2 extensions**: snow on ice, brine pockets, ridging, delta-Eddington shortwave, and melt ponds
- Sea-ice test matrix (`scripts/matrix/run_sea_ice_test_matrix.py`): 15 standard tests covering thermo, dynamics, transport, ITD, integration

### Lakes

- **Two-layer lake** with mixing and surface coupling (`coupler/lake/`)

### Coupler

- **Tile-based coupling** of ocean / sea-ice / land / lake with area-weighted blending of fluxes and surface state
- **Bulk flux**: COARE 3.0, Large & Yeager 2004, fixed-z₀
- **Surface albedo**: zenith-dependent ocean (Briegleb 1992), snow age decay, sea-ice temperature feedback
- **Surface energy and exchange** with conservation-aware accumulators

### Diagnostics

- **Energy budget**: column moist/dry static energy, TOA / surface flux tracking, dE/dt residual monitoring
- **Monthly means**: zonal-mean profiles, global-mean scalars, multi-year accumulation
- **Column integrals** (CWV, CIWV, MSE) and precision-drift monitors
- **CMIP6-style output**: CF-1.8 + CMIP6 DRS via `CFWriter` / `cmor_output.py` with **Amon, Lmon, Omon, Oyr, Ofx, SImon, SIyr** tables, overturning streamfunction, OSNAP transports, dianeutral mixing, and global ocean / sea-ice scalars

### External Forcing

- **GHG**: constant or time-varying (NetCDF), with CMIP6 experiment templates (piControl, historical, AMIP, 1pctCO2, SSP2-4.5, SSP5-8.5)
- **Ozone / aerosol / solar**: climatological or transient from files; CMIP6-shape loaders for `vmro3`, Kinne aerosol, MPI-M 14-band TSI, and CMIP6 volcanic AOD
- **AMIP CMIP6 deck** (`scripts/run/run_amip_cmip6_deck.py`): RRTMG + Sundqvist clouds + Sundqvist large-scale + SBM + Louis with the full transient stack on cubed-sphere and lat-lon production grids; Gaussian spectral and Voronoi/MPAS exercised by the dispatch smoke test
- **OMIP forcing** (`scripts/run/run_omip.py`)
- **Real topography / bathymetry** as above

### Data Assimilation

- **Variational DA** (`legoesm.da`): cost-function builder, control-vector handling, observation operators, preconditioning, incremental 4D-Var loop, NMC and gen-be background-error generators, and JIT-compiled minimizer — all built on `jax.grad` through the compiled segment kernel

### Machine Learning & Training

- **SFNO** (Spherical Fourier Neural Operator) blocks, spectral convolutions, normalization, and conservation-aware losses (`legoesm.ml`)
- **Joint ML / physics workflows**: trainable physics parameters with sigmoid constraints; dycore + SFNO coupling; spectral NeuralGCM training; SFNO + slab ocean S2S workflow; NeuralGCM + slab ocean S2S workflow (`legoesm.ml.s2s`)
- **ERA5 ingestion** to model state on any supported grid (`training/era5_to_state.py`)
- **Differentiable training driver** (`training/training_driver.py`) with non-donating segment kernels for AD, explicit `SegmentForcing` to avoid recompilation, and gradient checkpointing through `lax.scan`

### Drivers

- **`ModelDriver`** — single entry point for AMIP-style atmosphere-only simulations (`run_amip.py` is a thin CLI wrapper)
- **`CoupledESMDriver`** — fully coupled atmosphere / ocean / sea-ice / land / lake / carbon (`scripts/run/run_coupled.py`) with presets: `aquaplanet`, `slab_simple`, `slab_pft`, `slab_richards`, `slab_carbon`, `full_coupled`
- **`EarthSystemDriver`** — research orchestration for arbitrary component compositions
- **OMIP / centennial ocean driver** (`scripts/run/run_omip.py`) — multi-decade JRA55-do or idealized OMIP-2 spin-up across lat-lon, tripolar (eORCA1), cubed-sphere, and MPAS Voronoi grids, with auto-restart, AMOC / OSNAP / RPE tracking, and a `jra55_3way` run set (MPAS ico5 / ico6 / tripole eORCA1)
- **Single-column driver** (`scripts/matrix/run_scm_test_matrix.py`) — RCE, GABLS1, Ekman, Wangara, and oracle generation through one dispatcher; any-scheme × any-integrator swap matrix
- **Offline land driver** (`scripts/run/run_lmip.py`) — single-point multilayer land 10-year soil spin-up before ERA5 coupling
- **`PhysicsPipeline`** — radiation sub-cycling via `jax.lax.cond`, with cloud-radiation coupling and ML-physics dispatch
- **`DiagnosticCollector`** — energy budget, monthly means, snapshot history

### Parallelism

- **Canonical parallel runtime** `ParallelRuntime.create()`: serial, multi-device (SPMD), MPI, and hybrid execution under one API
- **Cubed-sphere sharding**: face-level (1/2/3/6 devices) and sub-face tiling (6k² for k≥2: 24, 54, 96, 150, …)
- **Lat-lon and level sharding**: domain decomposition by latitude or vertical levels
- **Voronoi**: recursive coordinate bisection + METIS partitioning with halo exchange
- **MPI**: mpi4jax-based 4D halo exchange (`pad_halo_4d`, single message per exchange) and reductions, with a `_sendrecv_vjp` `custom_vjp` wrapper that makes halo communication differentiable; `allreduce(SUM)` is AD-safe (max/min are diagnostic-only)
- **Ensemble parallelism**: `vmap` + `NamedSharding` with scan-based time integration and gradient checkpointing
- **Apple Silicon**: Metal/CPU hybrid routing — finite-volume solvers on Metal (`float32`); spectral solvers pinned to CPU for `float64`/`complex128`

### Validation Infrastructure

- **Test matrices**:
  - Atmosphere: `scripts/matrix/run_atmosphere_test_matrix.py` (Williamson 2/5/6, Jablonowski–Williamson + rotated DCMIP-2008 §4-1/§4-2, Held–Suarez ± topography, DCMIP 2012 §2-0-0 rest-with-topography, DCMIP transport, RCE, …) — selectable via `--family {sw,hydro,nh,climate,tracer,dcmip2008,dcmip2012,dcmip2016,hughes,all}`. Williamson CLI emits both PlateCarree u/v and native D-grid winds for plotting (issue #274)
  - Ocean: `scripts/matrix/run_ocean_test_matrix.py` (57/57 PASS across lat-lon, tripolar, cubed-sphere, MPAS Voronoi)
  - Sea ice: `scripts/matrix/run_sea_ice_test_matrix.py` (15 benchmark tests)
- **CFL-aware numerical-convergence tests + plotters**: term-by-term analytic shallow-water and ocean tests
- **Dycore validation catalog**: [`docs/validation/dycore_validation_catalog.md`](docs/validation/dycore_validation_catalog.md) — complete have/missing inventory against Hughes (2026) *"How to validate a 3D spherical dynamical core"* tutorial
- **Dycore progression suite** (`tests/validation/run_dycore_progression_suite.py`)
- **Ocean fidelity assessment harness** (`ocean/fidelity/`): Veros DINO / Eady adapters and cross-model comparison reports under `docs/ocean/fidelity/`
- **Distributed tests** including MPI differentiability (`tests/distributed/test_mpi_differentiability.py`)
- **Scaling benchmarks** (`scripts/bench/run_levante_gpu_scaling.py`, `scripts/bench/run_cpu_mpi_scaling.py`)
- **Source-guardrail harness** (static tripwires that verify the existing source obeys the project rules): ratchet audits (`tests/test_no_hardcoded_constants.py`, `tests/test_no_saturation_reimpl.py`), dispatch hardening (`tests/test_dispatch_hardening.py`), spec-first physics contracts (`tests/test_physics_contracts.py`), federation boundaries (`tests/test_import_boundaries.py`, `tests/test_federation_plan.py`), plus LIVE editor hooks in `.claude/hooks/`. Design: [`docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`](docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md)
- **Scientific validators** (`scripts/validate/*.py`, each `python scripts/validate/<name>.py`): convection/barotropic/ocean-SCM physics, `validate_federation_packaging.py`, and `visual_regression.py` (cube-imprint/edge artifacts — inspect the PNGs)
- **Adversarial-review agents** (user-triggered): Codex (`/codex:adversarial-review --wait` → fix → `/codex:review --wait`, iterate to clean) and specialized subagents in `.claude/agents/` (`physics-validator`, `lego-modularity-tester`, `dycore-tester`, `test-differentiability`, `test-scalability`, …)

### Running the checks

```bash
# Fast static guardrails (seconds, no GPU) — verify the existing source
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu pytest \
  tests/test_no_hardcoded_constants.py tests/test_no_saturation_reimpl.py \
  tests/test_dispatch_hardening.py tests/test_physics_contracts.py \
  tests/test_import_boundaries.py tests/test_federation_plan.py -q

# Full unit + guardrail suite
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu pytest tests/ -q

# Scientific test matrices (heavier — exercise numerics + conservation gates)
JAX_ENABLE_X64=1 python scripts/matrix/run_atmosphere_test_matrix.py
JAX_ENABLE_X64=1 python scripts/matrix/run_ocean_test_matrix.py
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/matrix/run_scm_test_matrix.py
JAX_ENABLE_X64=1 python scripts/matrix/run_sea_ice_test_matrix.py
JAX_ENABLE_X64=1 python scripts/matrix/check_conservation_all.py

# Dycore progression (Williamson / Galewsky / Jablonowski–Williamson / Held–Suarez)
python tests/validation/run_dycore_progression_suite.py

# Federation packaging (per-member wheels build + root-absent import)
python scripts/validate/validate_federation_packaging.py
```

> Full details — pytest tiers, the matrix framework, the guardrail harness, and the
> review agents — are in [`docs/validation/TESTING.md`](docs/validation/TESTING.md). On Apple Silicon set
> `JAX_PLATFORMS=cpu` (the Metal backend is broken); use `JAX_ENABLE_X64=1` for
> scientific/conservation runs. legoESM is developed and tested on Linux and macOS
> (POSIX) only — Windows is unsupported (bash scripts, SLURM, POSIX subprocess
> assumptions); use WSL2.

## Quick Start

```bash
# Install — legoESM is a uv workspace of independently-installable members
# (legoesm-core, -atmosphere, -ocean, ...), so the install path depends on your
# tool (see "Installing" below for why, and for single-component installs):
python -m venv .venv && source .venv/bin/activate
uv sync --extra dev                          # with uv (resolves the workspace natively)
# …or, pip-only (no uv) — the helper resolves the inter-member DAG locally:
python scripts/experiment/install_federation.py --all --extras dev

# Run Williamson Test Case 2 (cubed-sphere shallow water)
legoesm test williamson --case 2 --resolution 48 --days 5

# Run an atmosphere AMIP simulation
JAX_ENABLE_X64=1 python scripts/run/run_amip.py --grid-type cubed_sphere --resolution 16 --days 365

# Single-column radiative-convective equilibrium (issue #277)
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/matrix/run_scm_test_matrix.py rce --days 50

# Run a fully coupled simulation (slab ocean + bucket land)
JAX_ENABLE_X64=1 python scripts/run/run_coupled.py --preset slab_simple --days 365

# Run an OMIP-2 spin-up on the tripolar eORCA1 grid (30 years)
JAX_ENABLE_X64=1 python scripts/run/run_omip.py --grid tripole --days 10950

# Single-point multi-year multilayer land spin-up
JAX_ENABLE_X64=1 python scripts/run/run_lmip.py --lat 45.5 --lon -93.1 --days 3650

# Run the AMIP CMIP6 deck (transient GHG / ozone / aerosol / solar / volcanic)
JAX_ENABLE_X64=1 python scripts/run/run_amip_cmip6_deck.py

# Tests
JAX_ENABLE_X64=1 pytest tests/
```

## Installing

legoESM is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) of
independently-installable members (`legoesm-core`, `legoesm-atmosphere`,
`legoesm-ocean`, `legoesm-land`, `legoesm-ice`, `legoesm-coupler`, `legoesm-ml`,
`legoesm-tools`, and the root `legoesm` meta-package). Each member depends on the
others as ordinary distributions (`legoesm-core~=0.1.0`, …) that resolve to the
in-tree source **only via** `[tool.uv.sources]` (`workspace = true`).

**This is why a bare `pip install legoesm` (or `pip install ./packages/atmosphere`)
fails** with `Could not find a version that satisfies the requirement
legoesm-core~=0.1.0 … (from versions: none)`: plain pip ignores `[tool.uv.sources]`
and looks for the members on PyPI, where they are not published. You need either
`uv` (which understands the workspace) or the bundled helper (which resolves the
inter-member dependency DAG against the in-tree source instead of PyPI):

```bash
# With uv — resolves the whole workspace natively:
uv sync --extra dev                 # full dev install
uv pip install --package legoesm-ocean   # one component, standalone

# Pip-only (no uv) — scripts/experiment/install_federation.py resolves the DAG:
python scripts/experiment/install_federation.py --all --extras dev   # full dev install
python scripts/experiment/install_federation.py atmosphere           # one component (editable; pulls only core)
python scripts/experiment/install_federation.py ocean land ice       # several components
python scripts/experiment/install_federation.py atmosphere --extras ml   # component + an extra (pulls legoesm-ml)
python scripts/experiment/install_federation.py atmosphere --wheels  # non-editable, from a local wheelhouse
python scripts/experiment/install_federation.py atmosphere --dry-run # just print the pip command
```

Each Earth-system component (`atmosphere`/`ocean`/`land`/`ice`) is mutually
independent (import-linter contract #2), so any one installs standalone on top of
`legoesm-core` and runs a single column as its cheapest gradient-check harness.
The orchestration cluster (`coupler`/`ml`/`tools`) is a mutual cycle and installs
together. The helper computes this closure for you from the members'
`pyproject.toml` files. `scripts/validate/validate_federation_packaging.py` proves
the per-member wheels build and import root-absent.

## Defining New Experiments & Scripts

legoESM separates **what you run** (versioned experiment *templates* + the
reproducibility manifest) from **the code that runs it** (bucketed scripts).
Full guide: [docs/validation/TESTING.md](docs/validation/TESTING.md) §4 and
[config/templates/README.md](config/templates/README.md).

### A new experiment (template → run → reproduce)

Experiments are versioned YAML templates resolved by the harness — no editing a
mega-runner by hand:

```bash
# 1. (optional) check/stage any external data the template needs
python scripts/experiment/fetch_data.py check 2d/williamson2_sw

# 2. materialize a runnable dir from a template + dotted overrides + machine profile
python scripts/experiment/init_experiment.py 2d/williamson2_sw \
    --name w2hi --output-dir ./runs/w2hi \
    -o grid.resolution=96 -o time.duration_hours=240
#    -> writes config.yaml (resolved), run.sh (launcher), run.yaml (intent)

# 3. run it (config.yaml is fed to `legoesm run`)
cd ./runs/w2hi && bash run.sh

# 4. reproduce ANY past run bit-for-bit from its manifest
legoesm reproduce ./runs/w2hi/<output>/run_manifest.json --check
```

To **add a template**, drop a YAML under `config/templates/<category>/` with an
`experiment:` block (`tier` / `complexity` / `extent` / `maturity` / `data` /
`conservation_gates`) above a standard `legoesm run` config, then validate +
regenerate the run-status table:

```bash
python scripts/experiment/validate_templates.py --write-status   # -> project_status.md
```

The driver writes a `run_manifest.json` (resolved config + `state_digest` +
`git_hash` + jax/numpy versions) for every run, so reproducibility is built in —
do **not** add a parallel tag/provenance system.

### A new script

Put it in the right bucket — never at the `scripts/` root (which holds only
`__init__.py`):

- a production driver → `scripts/run/`; a benchmark → `scripts/bench/`; a
  plotter → `scripts/plot/`; a validator/intercomparison → `scripts/validate/`;
  data prep → `scripts/data/`; **anything throwaway/debug → `scripts/tmp/`**.
- Reference sibling scripts with an absolute import (`from scripts.<bucket>.<mod> import …`)
  and compute the repo root as `Path(__file__).resolve().parents[2]` (bucketed
  scripts are two levels below the root). `.sh` wrappers: `cd "$(dirname "$0")/../.."`.

### A new test-matrix case

Subclass `MatrixRunner` (component-agnostic framework in
`legoesm.experiments.matrix`); declare your `MatrixCase`s and chain the
conservation gates — the tier filtering, PASS/FAIL recording, `summary.json`,
and regression detection are inherited:

```python
from legoesm.experiments.matrix import MatrixRunner, MatrixCase, RunStatus, mass_gate, energy_gate

class OceanMatrix(MatrixRunner):
    component = "ocean"
    def build_cases(self):
        return [MatrixCase("ocean", "rest_state", "cubed_sphere", tier=1,
                           complexity="full_3d", resolution="C24", duration_days=5)]
    def run_case(self, case, *, quick, output_dir):
        ok, notes = True, ""
        ok, notes = mass_gate(ok, notes, volume_series, component="ocean")
        ok, notes = energy_gate(ok, notes, heat_series, component="ocean")  # routes to heat_rel_drift
        return (RunStatus.PASS if ok else RunStatus.FAIL), notes, {}

if __name__ == "__main__":
    raise SystemExit(OceanMatrix().main())   # --tier/--grid/--only/--quick/--list
```

Tag pytest tests with the tier ladder (`pytestmark = pytest.mark.tier1`) and use
the `conservation_gate` fixture so unit tests assert PASS/FAIL with the **same**
gates the matrix runners use.

## Platform Notes

### Compatibility Matrix

**Install minimum** (from `pyproject.toml`): Python ≥3.11, JAX ≥0.4.35, mpi4jax ≥0.8,<0.9 (optional), mpi4py ≥4.1,<5 (optional).

**Tested range** — the versions CI and benchmarks run against:

| Path | Hardware / Backend | MPI Runtime | Tested JAX | Tested mpi4jax | Status / Notes |
|---|---|---|---|---|---|
| Finite-volume dycores + ocean (single-process) | CPU (`jax` CPU backend) | N/A | `>=0.8,<0.10` | N/A | Regular unit/regression path |
| Finite-volume dycores + ocean (single-process) | Apple Silicon Metal (`jax-mps` / MLX, `JAX_PLATFORMS=mps`) | N/A | `>=0.8,<0.10` | N/A | FV solvers only (`float32`); no `float64` |
| Spectral solvers (atmosphere/ocean) | CPU (`JAX_PLATFORMS=cpu`) | N/A | `>=0.8,<0.10` | N/A | Requires `float64`/`complex128`; not mps-compatible |
| Distributed MPI halo/reductions | CPU + OpenMPI (`mpirun`) | OpenMPI 4.x/5.x | `>=0.8,<0.10` | `>=0.8,<0.9` | Validated with `mpirun -np 2/3/6` |
| Multi-device scaling suite | CPU/GPU (if available) | Optional | `>=0.8,<0.10` | `>=0.8,<0.9` (MPI mode) | `scripts/bench/run_levante_gpu_scaling.py` |

JAX versions outside the tested range may work but are not guaranteed. Versions below the install minimum will fail at `pip install`.

`legoesm.parallel.reductions` enforces MPI compatibility guardrails at runtime:
- **Hard error** for `mpi4jax<0.8` (incompatible token semantics).
- **Warning** for JAX or mpi4jax outside the tested range. Set `LEGOESM_MPI_STRICT_COMPAT=1` to promote the warning to a hard error.

Detailed runbook for real-hardware MPI / multi-GPU scaling:
- [docs/performance/REAL_HARDWARE_SCALING.md](docs/performance/REAL_HARDWARE_SCALING.md)

### Parallel Runtime & Supported Device Counts

The canonical entry point for all parallelism is `ParallelRuntime.create()`:

```python
from legoesm.parallel import ParallelRuntime
rt = ParallelRuntime.create(grid_type="cubed_sphere", grid_n=48)
```

**Supported cubed-sphere device/rank counts** (others are rejected with a precise error):
- **Face-only**: 1, 2, 3, 6
- **Sub-face tiling**: 6k² for k ≥ 2: 24, 54, 96, 150, 216, 294, 384, 600, …

Unsupported counts (4, 5, 7, 8, 12, 36, 48, …) raise `ValueError` with the nearest valid counts. There is no silent round-down.

**Execution modes**:

| Mode | Ranks | Devices/rank | Halo backend | Reduction backend |
|------|-------|-------------|-------------|-------------------|
| `serial` | 1 | 1 | local | local |
| `multi_device` | 1 | N | JAX SPMD | local |
| `mpi` | N | 1 | MPI | MPI |
| `hybrid` | N | M | MPI + JAX | MPI + JAX |

**Deprecated APIs**: `partition_state()` (zero-masked global arrays) emits `DeprecationWarning`. Use `ParallelRuntime.scatter()` / `.gather()` instead.

### Performance & Compile-Cache

- **Persistent JAX JIT cache** (issue #273) is enabled by default; the
  first segment compile is cached to disk and reused across runs.
  Override with `LEGOESM_JAX_CACHE_DIR=/path/to/cache`, or disable with
  `LEGOESM_JAX_CACHE_DISABLE=1`.
- **SPMD halo backend** (issue #275) is activated in the AMIP
  production profile for multi-device runs; single-process runs are
  unchanged.
- AMIP throughput target tracking continues under issue #273.

### Apple Silicon (Metal/MPS backend)

The **spectral solver** (Gaussian grid + spherical harmonic transforms) requires
`float64` and `complex128` arithmetic, which Apple's Metal backend does not support.

If you are on Apple Silicon and want to use the spectral solver, force the CPU backend:

```bash
JAX_PLATFORMS=cpu python your_script.py
```

The **finite-volume solvers** (cubed-sphere shallow water, hydrostatic primitive
equations, lat-lon C-grid) work on all backends including Metal, using `float32` precision.

To check your current JAX backend:

```bash
python -c "import jax; print(jax.default_backend())"
```

## Versioning

legoESM follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).
**Current release: `0.1.0` (legoESM v0.1).**

The version is **single-sourced and lockstep across the whole federation**:

- **One source of truth** — the literal lives in exactly one place, the root
  `pyproject.toml` `[project].version`. There is no second copy to drift.
- **Lockstep members** — all eight workspace packages
  (`legoesm-core`, `-atmosphere`, `-ocean`, `-land`, `-ice`, `-coupler`, `-ml`,
  `-tools`) share the same version and depend on each other with `~=0.1.0`, so a
  non-workspace `pip install legoesm` can never resolve a mismatched core. Bump
  them together.
- **Runtime resolution** — `legoesm.__version__` (and `legoesm._version`) reports
  the version of *the code actually executing*: it reads the source-tree
  `pyproject.toml` first (so an un-reinstalled checkout is honest), then falls
  back to installed package metadata. `legoesm --version` prints it.
- **Docs track MINOR** — the README banner, the Scientific Guide, and the
  Technical Documentation carry the `MAJOR.MINOR` (e.g. *v0.1*) via a single
  `\legoesmversion` macro per LaTeX document, kept in step with the package
  version on each release.

To cut a release: bump `version` in the root and all `packages/*/pyproject.toml`
(keep them identical), update the `\legoesmversion` macro in the two `docs/*.tex`
+ this banner, tag, and rebuild the PDFs.

`v0.1` is the initial public release: the differentiable core, all components,
the coupler, the federation packaging, and the tiered test/experiment harness.

## Documentation

- [docs/user-guide/getting_started.md](docs/user-guide/getting_started.md) — **Newbie onboarding guide** (start here)
- [CHANGELOG.md](CHANGELOG.md) — Release notes / what changed
- [SPECIFICATION.md](docs/science/specs/SPECIFICATION.md) — Full technical specification
- [docs/dev-notes/implementation_summary.md](docs/dev-notes/implementation_summary.md) — Comprehensive summary of implementations and tests
- [docs/validation/cmip_readiness.md](docs/validation/cmip_readiness.md) — CMIP production readiness checklist
- [docs/user-guide/amip.md](docs/user-guide/amip.md) — AMIP experiment guide
- [docs/user-guide/scm.md](docs/user-guide/scm.md) — Single-column model (SCM) guide
- [docs/science/ml_physics_parameterization.md](docs/science/ml_physics_parameterization.md) — Joint ML physics workflow and canonical moist run
- [docs/dev-notes/slab_s2s_documentation.md](docs/dev-notes/slab_s2s_documentation.md) — NeuralGCM/SFNO slab-coupled S2S workflow
- [docs/performance/REAL_HARDWARE_SCALING.md](docs/performance/REAL_HARDWARE_SCALING.md) — Multi-GPU/MPI scaling guide
- [docs/dev-notes/LATLON_CGRID_MIGRATION.md](docs/dev-notes/LATLON_CGRID_MIGRATION.md) — Lat-lon C-grid migration notes
- [docs/dev-notes/cubed_sphere_edge_artifacts.md](docs/dev-notes/cubed_sphere_edge_artifacts.md) — FV3-faithful cubed-sphere notes
- [docs/dev-notes/ocean_experiments_reference.md](docs/dev-notes/ocean_experiments_reference.md) — Ocean idealized-experiment reference
- [docs/ocean/fidelity/legoesm_vs_veros_v2.md](docs/ocean/fidelity/legoesm_vs_veros_v2.md) — legoESM ↔ Veros peer-comparison report

## Acknowledgments

legoESM bundles [jax-rrtmgp](https://github.com/climate-analytics-lab/jax-rrtmgp)
(Apache 2.0 license) for correlated-k radiation. jax-rrtmgp was developed by
Jeff Parker (Google), Duncan Watson-Parris (UCSD), and Juan Nathaniel (Columbia
University).

## License

legoESM is **source-available**, not OSI open source. It is released under the
[PolyForm Noncommercial License 1.0.0](./LICENSE): free for noncommercial use —
academic research, education, personal projects, and noncommercial use by
charitable, educational, public-research, and government organizations.

**Commercial use — including by revenue-generating organizations of any legal
form, for-profit or nonprofit — requires a separate paid commercial license.**
See [COMMERCIAL-LICENSE.md](./COMMERCIAL-LICENSE.md). Royalties are payable to
Pierre Gentine and Columbia University (IP jointly held by Celest.Science and
Columbia University).
