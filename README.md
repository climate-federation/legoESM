# legoESM 1.0

**A Differentiable Earth System Model in JAX**

legoESM is a fully differentiable Earth System Model spanning weather to
climate timescales. Built from scratch in JAX, it couples atmosphere, ocean,
land, sea ice, lakes and a tile-based surface coupler through one interface.
You can take `jax.grad` end to end through the coupled model, for variational
data assimilation, parameter estimation, sensitivity analysis and hybrid
AI–physics modelling. Dynamical cores, physics schemes, grids, vertical
coordinates and time integrators are swappable bricks, hence the "lego".

**Install** (Python ≥ 3.11):

```bash
pip install legoesm
```

This installs the model as a Python library (`import legoesm`). The `legoesm`
command-line test cases, experiment templates and the walkthrough below still
need a clone of this repository; [Getting started](#getting-started-about-one-hour)
sets one up with `uv`.

**New here? Follow [Getting started](#getting-started-about-one-hour) below.**
It takes you from `git clone` to your own 3-D atmosphere experiment, with the
results plotted, in well under an hour. The feature inventory and reference
material come after it.

## Getting started (about one hour)

Every legoESM command in this section was run on a fresh clone of
`main` on 2026-10-12. The machine was a 2-core, 3.9 GB slice of a Linux CPU
node (DKRZ Levante). The times quoted are from that run, and a laptop will
usually be as fast or faster. macOS, Windows/WSL2 and GPU setups were **not**
re-verified for this walkthrough.

### 1. What you need

- **Linux or macOS.** Windows is unsupported; use WSL2. The scripts assume
  bash and POSIX.
- **git** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**,
  the Python package manager. uv also provides Python: the walkthrough used
  the Python 3.14 that uv picked. legoESM needs Python ≥ 3.11.
- **About 2 GB of disk**: the clone plus `.venv` measured 1.7 GB, not counting
  uv's download cache.
- **About 4 GB of RAM.** The walkthrough ran in 3.9 GB, and the first
  experiment peaked at 1.5 GB. Two experiments run side by side in that 3.9 GB
  were killed for lack of memory.
- **Internet access** for the clone and the package download.
- **Nothing else.** The first experiment needs no GPU, no external data, no
  accounts or credentials and no machine configuration.

> **On an HPC login node?** Many centres forbid running software on login
> nodes. Get an interactive compute allocation first, for example
> `salloc --partition=interactive ...` on SLURM systems, and run the steps
> below inside it.

### 2. Install (≈ 4 min)

```bash
git clone https://github.com/climate-federation/legoESM.git   # ≈ 100 s
cd legoESM
uv sync                                                       # ≈ 2 min
source .venv/bin/activate
```

- `uv sync` creates `.venv/` itself, so you do not need `python -m venv`.
- The 2 minutes were measured with a warm uv download cache. A first-ever
  install also downloads JAX, SciPy and the rest, so it takes longer on a
  slow connection.
- Every later command assumes the environment is **activated**, so that
  `python` and `legoesm` come from `.venv`.

Check the install:

```bash
python -c "import legoesm, jax; print('jax', jax.__version__, 'backend', jax.default_backend())"
# -> jax 0.10.0 backend cpu        (the version may differ; "cpu" is expected without a GPU)
legoesm --help
# -> usage: legoesm [-h] {run,test,benchmark,reproduce,wizard} ...
```

### 3. Smoke test: a 5-day shallow-water run (≈ 1.5 min)

```bash
legoesm test williamson --case 2 --resolution 48 --days 5 --output results/w2_quick
```

This runs Williamson test case 2, a steady zonal flow whose exact solution is
known, on a C48 cubed sphere. **It worked if** the log ends like this
(your numbers may differ slightly):

```
INFO:   Step    720/720 (100.0%) | Mass err: ... | Energy err: ...
INFO: Completed in 26.3s (27 steps/s)
INFO: Error norms (height field):
INFO:   L1:   1.527382e-03
INFO:   L2:   8.432300e-03
INFO:   Linf: 1.900420e-01
INFO: Conservation:
INFO:   Mass:   initial=1.205409e+18, final=1.205408e+18, relative change=-1.03e-06
INFO:   Energy: initial=1.543599e+22, final=1.544007e+22, relative change=2.64e-04
INFO: Plots saved to results/w2_quick/
```

`results/w2_quick/` then holds five PNGs: `williamson2_height.png`,
`williamson2_conservation.png`, `williamson2_u_platecarree.png`,
`williamson2_v_platecarree.png` and `williamson2_dgrid_winds.png`. Without
`--output`, nothing is written to disk.

### 4. Your first experiment: Held–Suarez, 10 days (≈ 4–7 min)

Held–Suarez (1994) is the standard dry-dynamical-core climate benchmark: the
3-D hydrostatic atmosphere relaxed towards a zonally symmetric equilibrium
temperature, with surface friction. It needs no external data.
Experiments start from versioned **templates** in `config/templates/`. Create
a run directory from the template, shortening it from 100 to 10 days:

```bash
python scripts/experiment/init_experiment.py 3d_idealized/held_suarez \
    --name hs10 --output-dir runs/hs10 -o time.duration_hours=240
```

This writes three files:

- `runs/hs10/config.yaml`: the resolved configuration.
- `runs/hs10/run.yaml`: the template, overrides, machine profile and time.
- `runs/hs10/run.sh`: the launcher. It sets `JAX_PLATFORMS=cpu` (taken from
  the default machine profile, see `config/machines/`) and calls
  `legoesm run config.yaml`.

Run it, keeping a log:

```bash
cd runs/hs10
(set -o pipefail; bash run.sh 2>&1 | tee run.log)
echo "exit code: $?"
```

The first segment includes JIT compilation; it took 39–69 s in two runs.
After that you get one `Day N:` line per simulated day, each followed by a
`CFL max:` line. Two identical runs took 254 s and 425 s in total, depending
on how busy the node was.
**It worked if** the log ends like this:

```
INFO:   Day     10: T=289.7K, max_v=2.9m/s, ETA 0.0h
INFO:     CFL max: 0.01
INFO: Done: 224.8s wall time, status=COMPLETED
INFO:   Results saved to output/held_suarez
...
INFO: Simulation completed: COMPLETED
```

and the last line printed is `exit code: 0`. A blow-up or any other failure
makes `run.sh` exit non-zero. `pipefail` passes that status through `tee`, so
the subshell exits non-zero too.

Two things in that log are expected:

- **Mean temperature falls and the winds stay weak.** The run starts from rest
  and cools towards the Held–Suarez equilibrium. Jets and eddies take tens of
  days to develop. That is why the template's default length is 100 days, which
  its own comment calls the spin-up.
- **The "Energy Budget Summary" says `Status: CHECK`.** That check compares the
  column energy change with top-of-atmosphere radiation, and Held–Suarez has
  no radiation. Its residual is simply the Held–Suarez cooling: +177 W/m²
  against dE/dt of −177 W/m². The moisture budget reports `PASS`.
- **Start-up warnings.** The verified run printed four warnings before day 1
  and still completed normally:
  - `UserWarning: Explicitly requested dtype float64 ... truncated to dtype float32`
    (the template runs in float32).
  - `hybrid levels ... carry NEGATIVE layer mass for p_s < 660.2 hPa`. This
    case has no orography.
  - `dt=600s exceeds CFL limit. Reducing to dt=200s`, the driver's automatic
    time-step check.
  - `JAX x64 is DISABLED — the per-step solar clock ...`. This case has no
    radiation.

### 5. Look at the results

Everything is in `runs/hs10/output/held_suarez/`:

| File | Contents |
|---|---|
| `results.txt` | Run summary: grid, time step, wall and JIT time, final means, budgets |
| `timeseries.npz` | Daily global means (`T_atm`, `max_wind`, …) and temperature profiles |
| `snapshots.npz` | 2-D fields at day 5 and day 10 (cubed-sphere `(6, 36, 36)` arrays) |
| `experiment_config.json` | The full configuration the driver ran, with every default filled in |
| `run_manifest.json` | Reproducibility record: resolved config + hash, legoESM commit, Python/JAX/NumPy versions, final-state digest |

Plot the time series (still inside `runs/hs10`, with the environment active):

```bash
python ../../scripts/plot/plot_amip.py output/held_suarez --no-show
# -> Saved: output/held_suarez/diagnostics.png
#    Saved: output/held_suarez/profiles.png
```

The script is named for AMIP but reads any `ModelDriver` run's
`timeseries.npz`.

Optionally, rerun the experiment from its record and check that the final
state is bit-identical. On the same machine this took ≈ 8.5 min:

```bash
legoesm reproduce output/held_suarez/run_manifest.json --check
# -> reproduce --check: MATCH — bit-identical (state_digest 27cd6550...)
```

A bit-identical match is only expected on the same software, hardware and
device layout. See
[docs/architecture/reproducibility.md](docs/architecture/reproducibility.md).

### 6. Try something different

Return to the repository root (`cd ../..`) first. Change any configuration
field with `-o dot.path=value`. Every `-o` must change the resolved
configuration, otherwise `init_experiment` stops:

```bash
# Coarser grid (C24) for 5 days — took ≈ 2 min
python scripts/experiment/init_experiment.py 3d_idealized/held_suarez \
    --name hs_c24 --output-dir runs/hs_c24 \
    -o grid.resolution=24 -o time.duration_hours=120
(cd runs/hs_c24 && bash run.sh)

# A typo is caught before anything is written:
python scripts/experiment/init_experiment.py 3d_idealized/held_suarez \
    --name bad --output-dir runs/bad -o grid.resoluton=24
# -> ERROR: override(s) had no effect on the resolved config — likely a
#    misspelled or non-runtime dot-path ...: grid.resoluton=24
```

More things to try:

- **The full 100-day spin-up.** Drop the `-o time.duration_hours=240`. At the
  measured 19–34 s per simulated day on 2 cores, expect 30–60 min; this is an
  estimate, not timed.
- **Read the template** (`config/templates/3d_idealized/held_suarez.yaml`) to
  see which fields exist: grid, dycore, physics switches, time, output.
- **List every template** and whether it validates:
  `python scripts/experiment/validate_templates.py`. Prints one PASS/FAIL line
  per template; all 29 pass.
- **Other entry points:** the single-column model
  ([docs/user-guide/scm.md](docs/user-guide/scm.md)), AMIP
  ([docs/user-guide/amip.md](docs/user-guide/amip.md)), and
  [More example runs](#more-example-runs). Those commands were not re-timed
  for this walkthrough.
- **Gradients (the point of a differentiable model):** see *The
  differentiability story* in
  [docs/user-guide/getting_started.md](docs/user-guide/getting_started.md)
  and [docs/science/ml_physics_parameterization.md](docs/science/ml_physics_parameterization.md).
- **How the model is assembled:**
  [docs/architecture/composability.md](docs/architecture/composability.md) and
  [docs/user-guide/getting_started.md](docs/user-guide/getting_started.md).

### 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or `yaml`, `legoesm`) | The system `python` was used, not the environment's. Run `source .venv/bin/activate` in the repository root, or call `.venv/bin/python` directly. |
| `legoesm: command not found` from `bash run.sh` | `run.sh` calls `legoesm` from `PATH` (the default machine profile does not activate a venv). Activate `.venv` first. |
| `Could not find a version that satisfies the requirement legoesm-core~=1.0.0` | Plain `pip install` from a checkout cannot resolve workspace members not yet on PyPI. Use `pip install legoesm` for a release, `uv sync`, or the pip-only helper in [Installing](#installing). |
| Run killed: `Out Of Memory` / exit 137 | Not enough RAM. Run one experiment at a time, reduce `grid.resolution`, or ask for more memory in your allocation. |
| `ERROR: --output-dir runs/... exists and is not empty` | `init_experiment` never overwrites. Pick a new `--output-dir` or remove the old one. |
| `ERROR: resolved config is invalid: ... grid.vertical_coord must be one of ('sigma', 'hybrid', 'cam_l32'); got 'none'` | `none` is not a vertical coordinate. Shallow water does not run through `legoesm run`: launch a `2d/` template with `init_experiment.py` and `bash run.sh`, which call the matrix runner; see [Known issues](#known-issues-in-the-getting-started-path). |
| Spectral solver fails on Apple Silicon | Metal has no `float64`; run with `JAX_PLATFORMS=cpu`. Not re-verified here; see [Apple Silicon](#apple-silicon-metalmps-backend). |
| Run seems to hang at the start | The first segment includes JIT compilation (39–69 s for Held–Suarez C36 on 2 cores). On the CPU backend the compile cache is **off** by default, so every run compiles again; see the log's first line and [Performance & Compile-Cache](#performance--compile-cache). |

### Known issues in the getting-started path

Found while writing this walkthrough on 2026-10-12. None blocks the path
above.

- **Shallow water does not run through `legoesm run`.** The
  `2d/williamson2_sw` and `2d/williamson5_sw` templates therefore name the
  matrix cases (`init_experiment.py 2d/williamson2_sw ...` writes a `run.sh`
  that calls the matrix runner). Besides choosing the case (`setup.name`,
  `setup.grid`) and `output.path`, only `setup.duration_days`,
  `setup.resolution` and `setup.quick` can be set; the rest is the matrix
  case's. There is no `run_manifest.json` / `legoesm reproduce` for them.
- **Held–Suarez ran at dt=200 s** although its template says
  `dt_seconds: 600`. The driver's generic time-step check reduced it, with a
  `WARNING` in the log. The first line of `results.txt` reads
  "legoESM AMIP run" for every `ModelDriver` run, including this one.

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
- **Single-column model (SCM)** (`legoesm.atmosphere.forcing.scm.scm.SingleColumnModel`, `scripts/matrix/run_scm_test_matrix.py`): dycore-free driver that reuses the full physics factory for RCE, GABLS-style boundary-layer cases, parameterization integration tests, and any-scheme × any-integrator swap-matrix sweeps
- **Vertical coordinates**: pure sigma, hybrid sigma–pressure (L20–L60, sinh stretching), and the CAM 32-level hybrid table (`cam_l32`)
- **Physics packages** (each with a config NamedTuple, factory dispatch in `integration.py`, and direct unit tests):
  - **Radiation**: gray (Frierson-style) and RRTMGP correlated-k (LW + SW) with diurnal cycle, prescribed/transient ozone, aerosols (per-band SSA/asymmetry), solar TSI, and cloud–radiation coupling with McICA or maximum-random cloud overlap
  - **Convection**: Bechtold, Tiedtke, Kain–Fritsch, Zhang–McFarlane, Emanuel, Kuo, SBM, plus a deep convection adjustment (DCA) — sharing a unified mass-flux core with a stratospheric mass-flux gate
  - **Microphysics**: Kessler, Sundqvist large-scale condensation, Morrison double-moment (CAM6 MG2 or SAM flavour), Seifert–Beheng, Thompson, P3, SDM, fast SBM, plus an ML emulator path
  - **Cloud fraction**: Sundqvist, Xu–Randall, and CAM6 CLUBB (PDF liquid fraction + CAM ice fraction)
  - **Boundary layer / turbulence**: Louis, Holtslag–Boville, YSU, EDMF, TKE 1.5-order, MYNN-2.5, CLUBB-lite, prognostic CLUBB, Smagorinsky, surface-layer Monin–Obukhov, vertical diffusion, with PBL-height diagnosis
  - **Gravity-wave drag**: Lindzen, Hines, McFarlane, Rayleigh, E3SM/CAM (incl. frontogenesis source), prognostic spectral, ML emulator; `+`-joined compositions (e.g. `hines+mcfarlane`)
  - **Held–Suarez** dry forcing for benchmark runs
- **Production AMIP configuration**: the CAM6 physics suite (Zhang–McFarlane, prognostic CLUBB, MG2-style Morrison, RRTMG + McICA, McFarlane orographic GWD, CESM Large–Yeager fluxes, interactive multilayer land) on the MPAS grid at level 6 / L32 — `config/amip/amip_production.yaml`; see [docs/user-guide/amip.md](docs/user-guide/amip.md)

### Ocean

- **3D ocean dynamics**: split-explicit baroclinic / barotropic (cubed-sphere C-D, lat-lon C-grid, MPAS Voronoi, FC-Gram) with implicit-barotropic, BEBT-blended free-surface, MAXVEL clip, sponge relaxation, mass-conservation projection, and shared baroclinic helpers (EOS-pressure iteration, virtual-salt freshwater flux, implicit bottom-drag factor)
- **Grids**: lat-lon (regular + Mercator), tripolar (NEMO eORCA1 mesh_mask loader + tensor pole-fold halo), cubed-sphere C-D, MPAS Voronoi (LSQ edge-to-cell + Thuburn kite-area TRiSK); cross-grid metric-consistency test matrix at 57/57 PASS
- **Spectral ocean** (research path) and **SFNO learned ocean**
- **Ocean physics**: KPP / NEMO-style TKE (zdftke) / CATKE / Richardson / constant vertical mixing; Jayne–St-Laurent (2001) abyssal tidal mixing with tracer-mixing integration; harmonic, biharmonic, GM-Redi (cubed-sphere, lat-lon, MPAS variants), Visbeck adaptive-GM, Leith viscosity, and backscatter lateral mixing; convective adjustment (enhanced diffusion + plume); linear / quadratic bottom drag with implicit factor; bulk-formula / restoring / prescribed surface forcing; shortwave penetration (Jerlov)
- **Ice-shelf cavity coupling**: Holland & Jenkins (1999) three-equation basal-melt at the ice base, lat-lon C-grid and MPAS apply paths
- **External forcing**: Dai–Trenberth global river runoff (point→grid projection), OMIP-2 sea-surface salinity restoring + WOA SSS climatology loader, JRA55-do RYF preload (float32-safe for 32 GB GPUs), and an external `tau` / `q_net` / `sw_down` pathway for C-grid PE
- **Biogeochemistry**: abiotic carbon (DIC + ALK with carbonate equilibria and air-sea CO₂ flux) and an NPZD ecosystem
- **Idealized experiments suite** (`ocean/experiments/`): rest state, Eady / Phillips / baroclinic gyres, ACC channel, Drake/Stommel, lock exchange, overflow, baroclinic & barotropic wave, geostrophic adjustment, inertia–gravity wave, global overturning, Silvestri baroclinic jet, Munk, Held–Larichev, NeverWorld2-lite, ISOMIP+
- **Realistic geometry**: NetCDF bathymetry (ETOPO/GEBCO/ERDDAP) with bilinear regridding, Laplacian smoothing, MEO r-cap steepness limiter, polar-cap masking, flood-fill isolated-basin removal, and strait enforcement
- **Centennial spin-up library** (`ocean.spinup`): AMOC@26.5°N tracker, RPE / volume / heat / salt drift diagnostics, declarative `ConvergenceCriteria`, Bryan–Lewis (1984) distorted-physics accelerated protocol, and auto-restart discovery (`scripts/run/run_omip.py`)
- **Peer-comparison fidelity harness** (`ocean/fidelity/`): Veros DINO / Eady adapters, regridder, and `docs/ocean/fidelity/legoesm_vs_veros_v2.md`; NEMO term-by-term twins (`scripts/run/run_dino.py`) and a FESOM2 (fesom-jax) CORE2 arm (`scripts/run/run_fesom_core2.py`) for cross-model comparison
- **Simple ocean**: slab mixed-layer and two-layer (cubed-sphere and MPAS variants)

### Land Surface

- **Slab land**: energy balance + bucket hydrology + snow with stomatal conductance (Farquhar + Ball–Berry / Medlyn / Jarvis)
- **Multi-layer land**: Richards equation (mixed-form Picard) with 6 retention curves (Van Genuchten, Clapp–Hornberger, Brooks–Corey, Campbell, PDI, Lu) and Johansen thermal diffusion; the interactive land in production AMIP
- **Canopy schemes** (`--land-surface-scheme`): simple surface energy balance, two-leaf sun/shade canopy, and the CLM-ML multilayer canopy (vendored, differentiable), with per-column PFT and canopy interception
- **LMIP biophysics driver** (`scripts/run/run_lmip.py`): offline single-point or global land runs with calibrated PFT parameters
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
- **Production AMIP**: `scripts/run/run_amip.py --config config/amip/amip_production.yaml` plus the machine input paths of `config/amip/amip_production.sh` (Levante) or `config/amip/amip_production.ginsburg.sh` (Ginsburg)
- **AMIP forcing smoke deck** (`scripts/run/run_amip_smoke_deck.py`, not production): a C16 RRTMG + Sundqvist + Morrison + SBM + Louis stack on flat terrain (no land) with synthetic forcing by default, exercising the transient forcing plumbing on every grid
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
  - Atmosphere: `scripts/matrix/run_atmosphere_test_matrix.py` (Williamson 2/5/6, Jablonowski–Williamson + rotated DCMIP-2008 §4-1/§4-2, Held–Suarez ± topography, DCMIP 2012 §2-0-0 rest-with-topography, DCMIP transport, RCE, …) — selectable via `--family {sw,hydro,nh,climate,tracer,moist,dcmip2008,dcmip2012,dcmip2016,hughes,all}` or `--only {sw,hydro,nh,all}`. Williamson CLI emits both PlateCarree u/v and native D-grid winds for plotting (issue #274)
  - Ocean: `scripts/matrix/run_ocean_test_matrix.py` (lat-lon, tripolar, cubed-sphere, MPAS Voronoi)
  - Sea ice: `scripts/matrix/run_sea_ice_test_matrix.py` (15 benchmark tests)
- **CFL-aware numerical-convergence tests + plotters**: term-by-term analytic shallow-water and ocean tests
- **Dycore validation catalog**: [`docs/validation/dycore_validation_catalog.md`](docs/validation/dycore_validation_catalog.md) — complete have/missing inventory against Hughes (2026) *"How to validate a 3D spherical dynamical core"* tutorial
- **Ocean fidelity assessment harness** (`ocean/fidelity/`): Veros DINO / Eady adapters and cross-model comparison reports under `docs/ocean/fidelity/`
- **Distributed tests** including MPI differentiability (`tests/distributed/test_mpi_differentiability.py`)
- **Scaling benchmarks** (`scripts/bench/run_levante_gpu_scaling.py`, `scripts/bench/run_cpu_mpi_scaling.py`)
- **Source-guardrail harness** (static tripwires that verify the existing source obeys the project rules): ratchet audits (`tests/test_no_hardcoded_constants.py`, `tests/test_no_saturation_reimpl.py`), dispatch hardening (`tests/test_dispatch_hardening.py`), spec-first physics contracts (`tests/test_physics_contracts.py`), federation boundaries (`tests/test_import_boundaries.py`, `tests/test_federation_plan.py`), plus LIVE editor hooks in `.claude/hooks/`. Design: [`docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`](docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md)
- **Scientific validators** (`scripts/validate/*.py`, each `python scripts/validate/<name>.py`): convection/barotropic/ocean-SCM physics, `validate_federation_packaging.py`, and `visual_regression.py` (cube-imprint/edge artifacts — inspect the PNGs)
- **Adversarial-review agents** (user-triggered): Codex (`/codex:adversarial-review --wait` → fix → `/codex:review --wait`, iterate to clean) and specialized subagents in `.claude/agents/` (`physics-validator`, `lego-modularity-tester`, `dycore-tester`, `test-differentiability`, `test-scalability`, …)

### Running the checks

The test tools (`pytest`, `ruff`, …) are in the `dev` extra, which a plain
`uv sync` does not install: run `uv sync --extra dev` first.

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

# Dycore progression (shallow water -> hydrostatic -> non-hydrostatic ladder;
# --only sw|hydro|nh runs one rung, the default runs all three)
JAX_ENABLE_X64=1 python scripts/matrix/run_atmosphere_test_matrix.py --only sw

# Federation packaging (per-member wheels build + root-absent import)
python scripts/validate/validate_federation_packaging.py
```

> Full details — pytest tiers, the matrix framework, the guardrail harness, and the
> review agents — are in [`docs/validation/TESTING.md`](docs/validation/TESTING.md). On Apple Silicon set
> `JAX_PLATFORMS=cpu` (the Metal backend is broken); use `JAX_ENABLE_X64=1` for
> scientific/conservation runs. legoESM is developed and tested on Linux and macOS
> (POSIX) only — Windows is unsupported (bash scripts, SLURM, POSIX subprocess
> assumptions); use WSL2.

## Installing

legoESM is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) of
independently-installable members (`legoesm-core`, `legoesm-atmosphere`,
`legoesm-ocean`, `legoesm-land`, `legoesm-ice`, `legoesm-coupler`, `legoesm-ml`,
`legoesm-tools`, and the root `legoesm` meta-package). Each member depends on the
others as ordinary distributions (`legoesm-core~=1.0.0`, …) that resolve to the
in-tree source **only via** `[tool.uv.sources]` (`workspace = true`).

From PyPI, `pip install legoesm` (or one member, e.g. `pip install
legoesm-ocean`) installs released wheels. **From a checkout, a bare `pip install .`
(or `pip install ./packages/atmosphere`) fails** for any member version not yet
on PyPI, with `Could not find a version that satisfies the requirement
legoesm-core~=1.0.0 … (from versions: none)`: plain pip ignores `[tool.uv.sources]`
and looks for the members on PyPI. For a checkout you need either
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
python scripts/experiment/fetch_data.py check 3d_idealized/held_suarez

# 2. materialize a runnable dir from a template + dotted overrides + machine profile
python scripts/experiment/init_experiment.py 3d_idealized/held_suarez \
    --name hs_c24 --output-dir ./runs/hs_c24 \
    -o grid.resolution=24 -o time.duration_hours=120
#    -> writes config.yaml (resolved), run.sh (launcher), run.yaml (intent)

# 3. run it (config.yaml is fed to `legoesm run`)
cd ./runs/hs_c24 && bash run.sh

# 4. rerun a past run from its manifest; --check compares the final-state digest
legoesm reproduce output/held_suarez/run_manifest.json --check
```

To **add a template**, drop a YAML under `config/templates/<category>/` with an
`experiment:` block (`tier` / `complexity` / `extent` / `maturity` / `data` /
`conservation_gates`) above a standard `legoesm run` config, then validate +
regenerate the run-status table:

```bash
python scripts/experiment/validate_templates.py --write-status   # -> project_status.md
```

The production drivers (`legoesm run`, `run_amip.py`, `run_omip.py`,
`run_omip_core2.py`, `run_lmip.py`, `run_coupled.py`, `run_dino.py`) write a
`run_manifest.json` (resolved config + `state_digest` + legoESM commit +
jax/numpy versions); extend it rather than adding a parallel tag/provenance
system. Atmosphere (`ModelDriver`) runs also record every scheme config their
kernels run with (convection, microphysics, turbulence, gravity-wave drag,
radiation and clouds, land, forcing), including full overrides. `legoesm reproduce` refuses to rerun on
different code unless `--allow-version-drift` is given, and refuses in every
case when today's code does not rebuild the recorded configuration exactly or
resolves different physics. The ocean, land, DINO and coupled drivers record
their configs but `reproduce` does not rerun them yet, and a bitwise
`--check` match can only be expected on identical software, hardware and device
layout. See
[docs/architecture/reproducibility.md](docs/architecture/reproducibility.md)
for the gaps and the planned fixes.

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

## More example runs

These commands come from the previous Quick Start. They were **not**
re-run for the getting-started walkthrough above, and several need external
forcing data or many hours of compute. The production AMIP commands need the
machine input paths that the `config/amip/amip_production*.sh` launchers
provide.

```bash
# Install first: see Getting started above, or Installing below.

# Run Williamson Test Case 2 (cubed-sphere shallow water)
legoesm test williamson --case 2 --resolution 48 --days 5

# Run an atmosphere AMIP smoke test (bare-CLI default physics)
JAX_ENABLE_X64=1 python scripts/run/run_amip.py --grid-type cubed_sphere --resolution 16 --days 365

# Production AMIP (CAM6 suite on MPAS; needs the machine paths in amip_production.sh)
PY=.venv/bin/python DAYS=60 OUTDIR=results/amip_prod
source config/amip/amip_production.sh
"${PY}" -u scripts/run/run_amip.py --config config/amip/amip_production.yaml \
    "${AMIP_PATH_FLAGS[@]}" --days "${DAYS}" --output "${OUTDIR}"

# Same on Ginsburg (machine input paths from its launcher)
source config/amip/amip_production.ginsburg.sh
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --config config/amip/amip_production.yaml "${AMIP_PATH_FLAGS[@]}" --days 30

# Single-column radiative-convective equilibrium (issue #277)
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/matrix/run_scm_test_matrix.py rce --days 50

# Run a fully coupled simulation (slab ocean + bucket land)
JAX_ENABLE_X64=1 python scripts/run/run_coupled.py --preset slab_simple --days 365

# Run an OMIP-2 spin-up on the tripolar eORCA1 grid (30 years)
JAX_ENABLE_X64=1 python scripts/run/run_omip.py --grid tripole --days 10950

# Single-point multi-year multilayer land spin-up
JAX_ENABLE_X64=1 python scripts/run/run_lmip.py --lat 45.5 --lon -93.1 --days 3650

# Smoke-test the AMIP forcing plumbing (synthetic forcing, flat, not production)
JAX_ENABLE_X64=1 python scripts/run/run_amip_smoke_deck.py

# Tests
JAX_ENABLE_X64=1 pytest tests/
```

## Platform Notes

### Compatibility Matrix

**Install minimum** (from `pyproject.toml`): Python ≥3.11, JAX ≥0.5.3, mpi4jax ≥0.8,<0.10 (optional), mpi4py ≥4.1,<5 (optional).

**Tested range** — the versions CI and benchmarks run against:

| Path | Hardware / Backend | MPI Runtime | Tested JAX | Tested mpi4jax | Status / Notes |
|---|---|---|---|---|---|
| Finite-volume dycores + ocean (single-process) | CPU (`jax` CPU backend) | N/A | `>=0.8,<0.10` | N/A | Regular unit/regression path |
| Finite-volume dycores + ocean (single-process) | Apple Silicon Metal (`jax-mps` / MLX, `JAX_PLATFORMS=mps`) | N/A | `>=0.8,<0.10` | N/A | FV solvers only (`float32`); no `float64` |
| Spectral solvers (atmosphere/ocean) | CPU (`JAX_PLATFORMS=cpu`) | N/A | `>=0.8,<0.10` | N/A | Requires `float64`/`complex128`; not mps-compatible |
| Distributed MPI halo/reductions | CPU + OpenMPI (`mpirun`) | OpenMPI 4.x/5.x | `>=0.8,<0.10` / `>=0.10,<0.11` | `>=0.8,<0.9` / `>=0.9,<0.10` | Two generations, paired: legacy (jax 0.8–0.9 + mpi4jax 0.8) or FFI (jax 0.10 + mpi4jax 0.9); validated with `mpirun -np 2/3/6` |
| Multi-device scaling suite | CPU/GPU (if available) | Optional | `>=0.8,<0.10` | `>=0.8,<0.9` (MPI mode) | `scripts/bench/run_levante_gpu_scaling.py` |

JAX versions outside the tested range may work but are not guaranteed. Versions below the install minimum will fail at `pip install`.

`legoesm.parallel.reductions` enforces MPI compatibility guardrails at runtime:
- **Hard error** for `mpi4jax<0.8` (incompatible token semantics).
- **Warning** for JAX or mpi4jax outside the tested generations, including a cross pairing (e.g. mpi4jax 0.8 with jax 0.10). Set `LEGOESM_MPI_STRICT_COMPAT=1` to promote the warning to a hard error.

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

- **Persistent JAX JIT cache** (issue #273) is enabled by default on GPU/TPU;
  the first segment compile is cached to disk and reused across runs.
  On the **CPU backend it is off** unless `LEGOESM_ALLOW_CPU_COMPILE_CACHE=1`
  (the cache key is not CPU-microarchitecture-specific, so only opt in on a
  homogeneous CPU pool). Default location `$XDG_CACHE_HOME/legoesm/jit_cache`
  (else `~/.cache/legoesm/jit_cache`); override with
  `LEGOESM_JIT_CACHE_DIR=/path/to/cache`, or disable with
  `LEGOESM_JIT_CACHE_DIR=""`.
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
**Current release: `1.0.0` (legoESM 1.0).**

The version is **single-sourced and lockstep across the whole federation**:

- **One source of truth** — the literal lives in exactly one place, the root
  `pyproject.toml` `[project].version`. There is no second copy to drift.
- **Lockstep members** — all eight workspace packages
  (`legoesm-core`, `-atmosphere`, `-ocean`, `-land`, `-ice`, `-coupler`, `-ml`,
  `-tools`) share the same version and depend on each other with `~=1.0.0`, so a
  non-workspace `pip install legoesm` can never resolve a mismatched core. Bump
  them together.
- **Runtime resolution** — `legoesm._version.__version__` reports the version
  of *the code actually executing*: it reads the source-tree `pyproject.toml`
  first (so an un-reinstalled checkout is honest), then falls back to installed
  package metadata. `python -c "from legoesm._version import __version__; print(__version__)"`
  prints it.
- **Docs track MINOR** — the README banner, the Scientific Guide, and the
  Technical Documentation carry the `MAJOR.MINOR` (e.g. *1.0*) via a single
  `\legoesmversion` macro per LaTeX document, kept in step with the package
  version on each release.

To cut a release: bump `version` in the root and all `packages/*/pyproject.toml`
(keep them identical), update the `\legoesmversion` macro in the two `docs/*.tex`
+ this banner, tag, and rebuild the PDFs.

`1.0` is the first stable release: the differentiable core and every component,
each on the grid families it supports, the coupler, the CAM6-suite production
AMIP configuration on MPAS, the multilayer land with two-leaf (production) and
CLM-ML canopies, the NEMO/FESOM2
fidelity-matched ocean, multi-GPU/MPI execution, and the tiered
test/experiment harness. See [CHANGELOG.md](CHANGELOG.md).

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

legoESM is open source under the [MIT License](./LICENSE). Bundled third-party
components keep their own licenses (see their `LICENSE` files).
