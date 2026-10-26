# Configuration wizard

The legoESM **configuration wizard** is an interactive front-end that walks you
through the axes of a simulation — component, model type, physics-vs-ML, grid,
discretization, time integrator, duration, device, precision — and turns your
answers into a runnable bundle. It is the fastest way to get from "I want to run
*X*" to a launchable directory without hand-writing a config.

Source: `wizard.py` (the questionary terminal UI) and `wizard_core.py` (the pure,
unit-tested routing/validation half), both under `scripts/experiment/`.
Entry point: `legoesm wizard`.

```{note}
The wizard does **not** invent numerics. Every menu is read live from the model's
own registries (the driver `supported_matrix`, the dynamics/discretization option
lists, the integrator dispatch table, the coupled `PRESETS`, `config/templates/`,
`config/machines/`), and then gated against the target launcher's own argument
parser. Add a grid or a dynamical core to the model and the wizard offers it with
no edit; a model-side rename fails a unit test rather than silently emitting a
dead flag.
```

## Install

The wizard needs the optional `questionary` dependency, shipped as the `wizard`
extra:

```bash
pip install -e ".[wizard]"        # or, minimally:  pip install questionary
```

Without it, `legoesm wizard` exits with a one-line install hint instead of a
traceback.

## Run

```bash
legoesm wizard
```

or, equivalently, straight from the source tree:

```bash
python scripts/experiment/wizard.py
```

```{note}
**Apple Silicon.** The Metal JAX backend has no `float64` and crashes on import
for parts of this model, so pin the CPU backend before launching:

    JAX_PLATFORMS=cpu legoesm wizard

The wizard imports the model registries (and therefore JAX) lazily while building
the menus, so this matters even though you have not started a simulation yet.
```

Press <kbd>Ctrl-C</kbd> / <kbd>Esc</kbd> at any prompt to abort cleanly (nothing
is written).

## What it asks

The questions branch on your first two answers (**objective** and, for a
simulation, **component**). The full axis map:

| Axis | Choices |
|------|---------|
| **Objective** | `simulate` (process-based physics) · `train` (an ML emulator on ERA5) |
| **Component** | `atmosphere` · `coupled` (atmosphere + land/ocean/ice) · `ocean` |
| **Atmosphere model type** | `global` (idealized / template-driven) · `amip` (prescribed-SST) · `les_crm` (doubly-periodic plane) · `scm` (single column) |
| **Physics ↔ ML ladder** | process physics → **AIMIP** `classical` / `column_nn` / `sfno_physics` (ML-emulated physics) → `neural_gcm_spectral` (SFNO emulating dycore + physics — "ML everything") |
| **Simulation case** | a `config/templates/` template (shallow water, 3-D idealized, Held–Suarez, …); AMIP; an ocean-matrix case; an LES/CRM or SCM case |
| **Grid / discretization / integrator** | pruned live to the driver-supported `(model_type, discretization, grid)` triples; integrators gated to what the chosen path accepts |
| **Coupled preset** | the land-complexity ladder: `aquaplanet` → `slab_simple` → `slab_pft` → `slab_richards` → `slab_carbon` → `full_coupled` |
| **Duration** | hours (global atm) / days (coupled, ocean, training) |
| **Compute backend** | `cpu` · `gpu` · `mpi` (with rank count) |
| **Precision** | 32-bit or 64-bit (spectral/Gaussian paths force 64-bit) |
| **Machine profile** | auto-detect, or any profile under `config/machines/` |

```{note}
**Regional / limited-area** atmosphere is not offered — the model has no
limited-area core today; the "extent" choice is global vs doubly-periodic plane
(LES/CRM) vs single column. **DNS** is LES at fine resolution, not a separate
launcher.
```

## What it writes

After it prints the resolved plan (command, JAX platform, x64 flag, honest
caveats), the wizard writes a self-contained bundle to your chosen output
directory:

| File | Contents |
|------|----------|
| `config.yaml` | the resolved `legoesm run` config — **only** for the template-driven global-atmosphere path (other paths route to a dedicated script) |
| `run.sh` | the launcher: sets `JAX_PLATFORMS` / `JAX_ENABLE_X64`, activates the venv + SLURM directives from the machine profile, then runs the command |
| `wizard.yaml` | a **provenance stamp** — `wizard_schema_version`, `legoesm_version`, `git_commit`, `created_utc`, the machine, every selection, the launch command, and the caveats |

Run it (now from the wizard, or later by hand):

```bash
cd runs/<name>
bash run.sh
```

The provenance stamp means any bundle built today is always traceable to the exact
model commit it was built against.

## Worked examples

Each example shows the key answers and the command that lands in `run.sh`.

### Global idealized atmosphere (Held–Suarez)

```
objective         simulate
component         atmosphere
model type        global
template          3d_idealized/held_suarez
discretization    cdgrid
grid              cubed_sphere
integrator        ssp_rk3
resolution / nlev 16 / 20
duration          240 h
backend / bits    cpu / 32
→ legoesm run config.yaml          (config.yaml strict-validated before writing)
```

### AMIP (prescribed observed SST)

```
objective         simulate
component         atmosphere
model type        amip
dataset           cobe            (analytical needs no file; cobe/hadisst prompt for the SST path)
SST file          /data/COBE-SST2.nc
discretization    cdgrid
grid              cubed_sphere
integrator        ssp_rk3
days / dt         30 / 600
radiation         rrtmgp
backend / bits    cpu / 64
→ python scripts/run/run_amip.py --dataset cobe --grid-type cubed_sphere \
        --discretization cdgrid --resolution 16 --nlev 40 --days 30 --dt 600.0 \
        --time-integrator ssp_rk3 --radiation rrtmgp --output output \
        --forcing-path /data/COBE-SST2.nc
```

Convection and turbulence stay at `run_amip.py`'s defaults (`tiedtke` / `louis`); pass
extra flags to change them. The Gaussian grid forces 64-bit (spectral transform).

### AIMIP — ML-emulated physics

```
objective         train
training mode     aimip
variant           sfno_physics      (classical / column_nn / sfno_physics)
smoke test        yes
backend           gpu
→ python scripts/run/run_aimip.py \
        --suite <repo>/config/aimip/aimip_suite.yaml --variants sfno_physics --smoke
```

This **trains** the variant over the shipped `config/aimip/` suite (base
`config/aimip/aimip_era5.yaml`) and needs an ERA5 cache. Its `run.sh` launches
from the repo root so the suite's relative paths resolve; outputs land under the
suite's `output_dir` (`results/aimip_*`), not the bundle dir. x64 is forced.

### Ocean

```
objective         simulate
component         ocean
case              barotropic_double_gyre
grid              latlon_regional   (gated per case — gyres/ACC/Eady are regional/channel-only)
quick             yes
backend / bits    cpu / 64
→ python scripts/matrix/run_ocean_test_matrix.py \
        --only barotropic_double_gyre --grid latlon_regional --output output --quick
```

The grid menu is restricted to the grids the ocean matrix actually runs for the
chosen case, so the emitted command always hits at least one test (no silent
no-ops). The full case list: `python scripts/matrix/run_ocean_test_matrix.py --list`.

### Coupled (land complexity ladder)

```
objective         simulate
component         coupled
preset            slab_richards     (the --preset axis IS the land ladder)
resolution / nlev 16 / 20
days              30
radiation         gray
backend / bits    cpu / 64
→ python scripts/run/run_coupled.py --preset slab_richards --resolution 16 \
        --nlev 20 --days 30 --radiation gray --output output
```

### LES / CRM and SCM

- **LES / CRM** (doubly-periodic plane): pick `rising_thermal`, `rcemip`, or an
  ABL case (`neutral` / `ekman` / `gabls1` / `wangara`) → routes to the bespoke
  plane scripts (`run_plane_rising_thermal.py`, `run_rcemip_plane.py`,
  `run_les_plane.py`).
- **SCM** (single column): pick `rce` / `gabls1` / `ekman` / `wangara` → routes to
  `run_scm_test_matrix.py`.

### Training the spectral neural GCM

```
objective         train
training mode     neural_gcm_spectral
n_max / n_levels  42 / 10
epochs / lr       50 / 3e-4
backend           gpu
→ python scripts/run/train_neural_gcm_spectral.py --n-max 42 --n-levels 10 \
        --epochs 50 --lr 0.0003 --train-days 365 --year 2015 --cache-dir data/era5_cache
```

ERA5 must be staged first (a Zarr cache at `--cache-dir`, or network access to the
WeatherBench2 bucket). x64 is forced (spectral).

## How it stays in sync with the model

Two mechanisms keep the wizard from drifting as the model evolves:

1. **Live derivation.** Option lists are read from the model registries at call
   time, not hard-coded.
2. **Stamped provenance + drift tests.** `wizard_core.WIZARD_SCHEMA_VERSION` is
   recorded in every bundle, and `tests/unit/test_wizard_core.py` asserts every
   curated menu is a subset of the live registries **and** of each target
   launcher's own argument choices. A model-side rename or a launcher arg change
   fails a test instead of producing a command the launcher rejects.

If you add a new wizard branch, gate every emitted flag against the target
script's argument parser (not just the model registries), shell-quote argv via
`wizard_core._cmd`, and add a drift test — see the existing AMIP/ocean gating.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `the wizard needs the 'questionary' package` | `pip install -e ".[wizard]"` |
| Import crash on Apple Silicon | Prefix `JAX_PLATFORMS=cpu` (Metal has no float64) |
| `--output-dir … exists and is not empty` | Choose a fresh output directory (the wizard never overwrites a populated dir) |
| `could not build a launch plan: …` | An inconsistent answer set (e.g. an unsupported discretization/grid pair); the message names the conflict |
| AMIP `--forcing-path required` at run time | Pick `analytical` SST, or supply the SST file when prompted for `cobe`/`hadisst` |
