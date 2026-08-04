# LMIP biophysics runbook

**What this driver is** : a global forced land mode configuration, biophysics-only
(no carbon), driven by real CRU-JRA reanalysis or a synthetic fallback on NSF_NCAR's Derecho.
Multi-layer soil, Richards hydrology, single-bulk snow, prescribed seasonal LAI
from surfdata, two-leaf canopy or `simple_seb` surface scheme.

---

## Prerequisites (one-time per machine)

### Python env (Mac / login node)
```bash
conda create -n legoesm-lmip python=3.12 -y
conda activate legoesm-lmip
cd $LEGOESM_ROOT
python scripts/experiment/install_federation.py --all --extras dev
pip install netcdf4 gcsfs
python -c "import jax, xarray, netCDF4, legoesm; print('OK')"
```

### GPU env on Derecho (separate env — do not overwrite `legoesm-lmip`)
```bash
conda create -n legoesm-gpu python=3.12 -y
conda activate legoesm-gpu
python scripts/experiment/install_federation.py --all --extras dev
pip install --upgrade "jax[cuda12]" netcdf4 gcsfs
python -c "import jax; print(jax.devices())"       # must list a GpuDevice
```

### Data
```bash
# On Derecho, from the repo root:
./scripts/data/download_lmip_data.sh --year 1920
./scripts/data/download_lmip_data.sh --year 1921
./scripts/data/download_lmip_data.sh --year 1922
# … one call per year you want to force with
```
This symlinks CRU-JRA `Solr/Prec/TPQWL` from glade into `data/crujra/`, and
downloads the CLM5 surfdata from Zenodo into `data/legoesm_surfdata_c250617.nc`.

---

## Quick start — 3 commands

```bash
# 1. Instantiate an experiment from a template.
python scripts/run/init_experiment.py biophysics/smoke_4deg \
    --name mac_smoke --output-dir ./mac_smoke

# 2. Look at what was created.
ls mac_smoke/
#   run.yaml         — declaration (template + overrides)
#   config.yaml      — resolved config (what the driver reads)
#   experiment.tag   — INI provenance (legoESM SHA, Python/JAX versions)
#   run.sh           — one-shot wrapper

# 3. Run it.
cd mac_smoke && bash run.sh
```

Output: `lmip_biophys.monthly.nc`, `lmip_biophys.monthly_state.nc`,
`restart_<YEAR>_d<DDD>h<HH>.npz` — one restart file per run, model-time-stamped.

---

## Shipped templates

All under `templates/land/biophysics/`:

| Template | Grid | Physics | Forcing | Cost |
|---|---|---|---|---|
| `smoke_test` | 4×8 | simple_seb + constant | synthetic | seconds — hermetic test |
| `smoke_4deg` | 45×90 (~4°) | simple_seb + constant | CRU-JRA 30-day | minutes on CPU |
| `spinup_5year` | 90×180 (~2°) | two_leaf_canopy + MOST | CRU-JRA 5-year | ~2–4 h A100 |
| `lmip_canopy_10yr` | 90×180 (~2°) | two_leaf_canopy + MOST + **single** snow + freeze/thaw | CRU-JRA 10-year | **~8 h A100** — the production block |

`lmip_canopy_10yr` is the **reference production config** and the reusable
**10-year increment** for the spin-up → validation workflow below: override
`forcing.year_start`/`year_end` (and `restart.from`) with `-o` per increment. Its
10-year span keeps `time.n_steps = 87600`, so you never override `n_steps`.

Pick the one closest to your target, override individual fields with `-o`.

---

## Shipped machine profiles

All under `config/machines/`:

| Profile | Scheduler | Env default | Resources |
|---|---|---|---|
| *(none)* | none — plain `bash run.sh` | `sys.executable` at init | — |
| `derecho` | PBS | `${LEGOESM_CONDA_ENV:-legoesm-lmip}` | `1:ncpus=8:mem=32GB`, 2 h |
| `derecho_gpu` | PBS | `${LEGOESM_CONDA_ENV:-legoesm-gpu}` | `1:ncpus=32:mem=64GB:ngpus=1`, 4 h |

Pick one with `--machine <name>`. **The PBS project account (`-A`) is NEVER
emitted into `run.sh`** — supply it on the qsub line:

```bash
qsub -A ACCOUNT run.sh
```

Add a new profile by dropping a YAML file into `config/machines/`; the flag
picks it up by filename.

---

## Overriding config values (`-o dot.notation=value`)

Values are YAML-parsed, so:

```bash
python scripts/run/init_experiment.py biophysics/spinup_5year \
    --name my_run --output-dir $SCRATCH/my_run --machine derecho_gpu \
    -o forcing.year_start=1980 \
    -o forcing.year_end=1984 \
    -o physics.bulk_scheme=constant \
    -o time.dt=1800 \
    -o time.n_steps=87600           # 5 years at 30-min dt
```

The full schema is in `packages/land/legoesm/land/lmip_config.py`. Anything
listed there can be overridden.

**Rules the validator enforces:**

- `physics.surface_scheme = simple_seb` **requires** `bulk_scheme = constant`.
  (simple_seb + MOST is numerically unstable — the pair is rejected at
  config-load time.)
- `forcing.year_end >= forcing.year_start`
- `output.tapes` must have at least one tape

---

### The restart file
Model-time-stamped: `restart_<YEAR>_d<DDD>h<HH>.npz` — chronological on `ls`.
Contains the full `MultiLayerLandState` + INI-blob metadata (`land_mode`,
`t_end_s`, `n_steps_completed`, config snapshot). A multi-year run writes ONE per
completed year (a resume trail); a single-year run writes one at the end.

### `experiment.tag`
INI-format provenance. The minimum you need for reproducibility months later:

```ini
[legoESM]
commit = <SHA>          # git checkout this to reproduce

[reproducibility]
python_version = 3.12.13
jax_version = 0.10.1
config_hash = sha256:…  # dedupe against other runs
```

### `run.yaml` vs `config.yaml`
- **`run.yaml`** — declaration only (template name + `-o` overrides). *What you
  meant to run.*
- **`config.yaml`** — fully resolved. *What the driver actually ran.*

Path fields in `config.yaml` are made absolute at init time (relative to your
CWD when you ran `init_experiment.py`), so the experiment dir is portable —
move it, `bash run.sh` still works.

---

## Chaining runs across years (restart-based spin-up)

Every run auto-saves its end state to `restart_<YEAR>_d<DDD>h<HH>.npz`. Chain
years by pointing the next run's `restart.from` at that file:

```bash
# Year 1: cold start
python scripts/run/init_experiment.py biophysics/spinup_5year \
    --name spinup_1920 --output-dir $SCRATCH/lmip/spinup_1920 \
    --machine derecho_gpu \
    -o forcing.year_end=1920 -o time.n_steps=8760

qsub -A $ACCOUNT $SCRATCH/lmip/spinup_1920/run.sh
# … wait for it to finish, then:

ls $SCRATCH/lmip/spinup_1920/restart_*.npz
# e.g. restart_1921_d000h00.npz

# Year 2: warm-start
python scripts/run/init_experiment.py biophysics/spinup_5year \
    --name spinup_1921 --output-dir $SCRATCH/lmip/spinup_1921 \
    --machine derecho_gpu \
    -o forcing.year_start=1921 -o forcing.year_end=1921 -o time.n_steps=8760 \
    -o restart.from=$SCRATCH/lmip/spinup_1920/restart_1921_d000h00.npz

qsub -A $ACCOUNT $SCRATCH/lmip/spinup_1921/run.sh
```
---

**Data prerequisite:** stage CRU-JRA for every year you run (1975-2014 here):
```bash
for y in $(seq 1975 2014); do ./scripts/data/download_lmip_data.sh --crujra-only --year $y; done
```
A missing year fails fast (`total_steps != n_steps` → `SystemExit`).

---

## Reproducing an old run

```bash
cd <experiment_dir>
cat experiment.tag                                          # find the [legoESM] commit
git -C $LEGOESM_ROOT checkout <commit>
bash run.sh                                                 # or qsub -A <ACCT> run.sh
```

The `run.sh` embeds absolute paths and the exact driver invocation, so the same
experiment reproduces bit-identical (up to GPU nondeterminism) on the same
machine + env.

---

## Where things live in the repo

```
templates/land/biophysics/          — YAML templates (edit these to add experiments)
config/machines/                   — YAML machine profiles (mac, derecho, derecho_gpu)
scripts/run/
  init_experiment.py                — template → experiment directory
  run_lmip_biophys.py               — the driver (`--config PATH --output-dir DIR`)
packages/land/legoesm/land/
  lmip_config.py                    — schema + validator + override system
  forcing/cru_jra.py                — CRU-JRA reader + 6h→dt disaggregation
  output_tapes.py                   — CLM-style history tape aggregation
  restart.py                        — .npz state save/load
docs/land/
  lmip_biophys_runbook.md           — this file
```
