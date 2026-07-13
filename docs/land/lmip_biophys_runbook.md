# LMIP biophysics runbook

**Audience:** anyone (you, a teammate, future-you) who wants to run a
CRU-JRA-forced land biophysics simulation with legoESM — locally or on Derecho —
without reading source code.

**What this driver is** (and isn't): a global forced land model, biophysics-only
(no carbon), driven by real CRU-JRA reanalysis or a synthetic fallback.
Multi-layer soil, Richards hydrology, single-bulk snow, prescribed seasonal LAI
from surfdata, two-leaf canopy or `simple_seb` surface scheme. **Cold-region
cells (polar / high-altitude / boreal winter) are known to numerically diverge**
under this configuration — see "Known limitations" at the end. Physics fixes are
a separate workstream.

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

Pick the one closest to your target, override individual fields with `-o`.

---

## Shipped machine profiles

All under `configs/machines/`:

| Profile | Scheduler | Env default | Resources |
|---|---|---|---|
| *(none)* | none — plain `bash run.sh` | `sys.executable` at init | — |
| `derecho` | PBS | `${LEGOESM_CONDA_ENV:-legoesm-lmip}` | `1:ncpus=8:mem=32GB`, 2 h |
| `derecho_gpu` | PBS | `${LEGOESM_CONDA_ENV:-legoesm-gpu}` | `1:ncpus=32:mem=64GB:ngpus=1`, 4 h |

Pick one with `--machine <name>`. **The PBS project account (`-A`) is NEVER
emitted into `run.sh`** — supply it on the qsub line:

```bash
qsub -A UYAL0053 run.sh
```

Add a new profile by dropping a YAML file into `configs/machines/`; the flag
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

## Understanding output

### The tape NetCDFs
Each `output.tapes` entry produces its own file: `lmip_biophys.<tape_name>.nc`.
Shipped defaults: `monthly` (time-mean fluxes) + `monthly_state` (instantaneous
soil snapshot). Layout is `(time, lat, lon)` for latlon grids.

**Annual files (multi-year runs).** A run spanning more than one year ALSO writes
one file per completed year — `lmip_biophys.<tape_name>.<year>.nc` — flushed right
after that year's scan, together with a resumable restart. So if a job hits its
wall-clock limit, every FINISHED year is already on disk (and resumable via
`--restart-from` the last year's `.npz`); you don't lose the whole run. The
combined `lmip_biophys.<tape_name>.nc` is still written at the end for a run that
finishes. `ncrcat lmip_biophys.monthly.*.nc combined.nc` stitches the annual files
if you only have those.

```python
import xarray as xr
ds = xr.open_dataset("lmip_biophys.monthly.nc")
ds["T_sfc"].isel(time=0).plot(robust=True)     # first month
```

Variables available (defaults; add more via `output.tapes[*].vars` in the YAML):

| Variable | Units | Averaged? |
|---|---|---|
| `T_sfc` | K | mean |
| `shflx`, `lhflx` | W/m² (positive up) | mean |
| `runoff` | kg/m²/s | mean |
| `precip` | kg/m²/s | mean |
| `LAI` | m²/m² | mean |
| `albedo` | [0, 1] | mean |
| `GPP` | gC/m²/day (canopy gross primary production; 0 for simple_seb) | mean |
| `ET` | mm/day (latent-heat-equivalent evapotranspiration, lhflx / L_v) | mean |
| `transp` | mm/day (canopy transpiration, LE_canopy / L_v; canopy only) | mean |
| `soil_evap` | mm/day (ground evaporation, LE_soil / L_v; canopy only) | mean |
| `Rnet` | W/m² (net radiation into surface = sw_down·(1−α) + lw_down − lw_up) | mean |
| `reverted` | 0–1 (per-cell fraction of steps the NaN-revert guard fired — a diverging-cell map) | mean |
| `T_soil_top`, `theta_soil_top`, `snow_depth` | K, m³/m³, kg/m² | inst |

**Diverging cells (boreal/Arctic).** Columns are independent, so a cell whose
state goes non-finite is reverted to its previous step (its fluxes masked to NaN)
instead of poisoning the run — the run finishes and `PASS`es. Two diagnostics:
the `reverted` tape var above, and `lmip_biophys.reverts.nc` (per-cell count of
reverted steps). A non-zero count means the physics diverged there; see
`docs/land/boreal_nan_diagnosis_plan.md`.

`transp + soil_evap ≈ ET` (they partition total evapotranspiration); the split is
the key diagnostic for the #730 over-transpiration calibration.

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

qsub -A UYAL0053 $SCRATCH/lmip/spinup_1920/run.sh
# … wait for it to finish, then:

ls $SCRATCH/lmip/spinup_1920/restart_*.npz
# e.g. restart_1921_d000h00.npz

# Year 2: warm-start
python scripts/run/init_experiment.py biophysics/spinup_5year \
    --name spinup_1921 --output-dir $SCRATCH/lmip/spinup_1921 \
    --machine derecho_gpu \
    -o forcing.year_start=1921 -o forcing.year_end=1921 -o time.n_steps=8760 \
    -o restart.from=$SCRATCH/lmip/spinup_1920/restart_1921_d000h00.npz

qsub -A UYAL0053 $SCRATCH/lmip/spinup_1921/run.sh
```

Or drive multiple years in a **single** job — the driver now runs a chunked scan
(one year at a time internally), so `year_end - year_start > 0` works within one
PBS submission as long as **one year's forcing fits in device memory** (~16 GB
float64 at 2° hourly, well within a 40 GB A100).

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

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ERROR: CRU-JRA forcing not staged for the requested years:` | Missing year on disk | Run `./scripts/data/download_lmip_data.sh --year YYYY` for each listed year |
| `ERROR: per-year forcing pytree (X GiB) exceeds the soft budget (24 GiB).` | Grid × dt combo too big for GPU | Coarser grid (`smoke_4deg`), larger `time.dt`, or CPU with more `mem=…` |
| `ValueError: physics: simple_seb + MOST is numerically unstable` | Override picked an incompatible pair | Use `two_leaf_canopy` + MOST, or `simple_seb` + `constant` |
| `RuntimeError: Could not initialize backend 'gpu'` | `jax[cuda12]` not installed in the env | `conda activate legoesm-gpu && pip install --upgrade "jax[cuda12]"` |
| `FileNotFoundError: 'config.yaml'` on PBS | `run.sh` used `dirname $0` (spool path); needs current version | `git pull` — fixed in `6b2f6458e`; re-init the experiment |
| `NaN final T_soil_top over land: N -> FAIL` (small N) | Cold-region cells diverge (see limitations below) | Currently accept as a known gap; physics fix is next workstream |

---

## Transient land-use / land-cover change (LULCC)

LULCC is an **option**: the per-column PFT fractions evolve year-by-year from a
reconstruction (re-weighting LAI, canopy params, and albedo each step), with
optional E_LUC land-use-change carbon booked as a post-run diagnostic. The
transient-cover engine (`make_step_land_params_updater` + per-step `year`
threading) is the same machinery the coupled/AMIP path uses; it keys off the
**surfdata's number of cover years**, so LULCC is enabled by handing the driver a
*transient* surfdata (multi-year `pft_frac`) rather than a runtime switch.

To turn it on:

1. **Bake a transient surfdata** from a reconstruction (HYDE / Pongratz / KK10):
   ```bash
   python scripts/data/build_anthropogenic_surfdata.py --dataset hyde \
     --base data/legoesm_surfdata_c250617.nc \
     --year-start 1920 --year-end 1929 \
     --out data/legoesm_surfdata_hyde_1920-1929.nc
   ```
   LUH2/LUH3 (native **gross** transitions → best E_LUC fidelity) go through
   `build_luh2_transient_surfdata` instead.
2. **Declare it in the config** (see template `biophysics/lmip_canopy_lulcc`):
   ```yaml
   surfdata:
     path: data/legoesm_surfdata_hyde_1920-1929.nc
     land_cover_dataset: hyde          # clm5 (static) | luh2 | luh3 | hyde | pongratz | kk10
   land_use_change:
     scheme: bookkeeping               # writes <output>.eluc_annual.txt; omit to skip
   ```
   The cover years must span the forcing years.

The run banner prints `cover=transient LULCC ON (hyde, N cover years Y0-Y1)` and
`E_LUC=on`. **Fail-fast guard:** declaring a reconstruction (`land_cover_dataset`
≠ `clm5`) while pointing `surfdata.path` at a single-year surfdata is a hard error
(it would silently apply no land-use change) — build the transient surfdata first.
`land_cover_dataset: clm5` = static cover (the default, byte-identical to a
non-LULCC run).

## Known limitations (current infrastructure phase)

These are physics gaps, not code bugs. The next physics workstream will address
them.

- **No soil freeze/thaw.** Water is always liquid; sub-freezing soil energy
  budget can drift. Manifests as Antarctica / boreal-winter cell divergence.
- **Single-bulk snow, no cap.** Snow depth can grow unrealistically deep in
  interior Antarctica.
- **No dedicated glacier tile handling.** Ice-covered cells run as
  "soil-with-LAI=0" — the two-leaf-canopy Newton doesn't converge cleanly on
  those.

Typical impact on a real 1-year 2° `spinup_5year` run: ~54 % of land cells stay
finite through December; ~46 % (dominated by Antarctica + boreal winter) go
NaN. The physics of the surviving cells looks reasonable (T_sfc 227–311 K,
seasonal cycle, LAI 0–6).

---

## Where things live in the repo

```
templates/land/biophysics/          — YAML templates (edit these to add experiments)
configs/machines/                   — YAML machine profiles (mac, derecho, derecho_gpu)
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
  lmip_s3_scope.md                  — dev-facing scope + phase history
```
