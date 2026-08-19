# LMIP biophysics on Derecho — quick runbook

Global biophysics-only LMIP on Derecho (branch `lmip-biophys-stable`).  Runs
`scripts/run/run_lmip_biophys.py` (config-driven) against the real CRU-JRA
reanalysis on glade + the CLM5 surfdata hosted on Zenodo.

The default job is the **`biophysics/lmip_canopy_4deg_smoke`** template — a ~4°
global shakedown (two-leaf canopy + MOST + soil freeze/thaw ON), the small-first
test before scaling to the 2° `lmip_canopy_10yr` production spin-up.

## One-time setup

```bash
# 1. clone + branch
cd /glade/work/$USER          # or wherever you keep code
git clone https://github.com/climate-federation/legoESM.git
cd legoESM
git checkout lmip-biophys-stable

# 2. Python env (federation multi-package workspace under plain pip)
module load conda
conda create -n legoesm-lmip python=3.12 -y
conda activate legoesm-lmip
python scripts/experiment/install_federation.py --all --extras dev
pip install netcdf4 gcsfs
python -c "import jax, xarray, netCDF4, legoesm; print('env OK')"
```

## Each run

```bash
cd $LEGOESM_DIR
conda activate legoesm-lmip
git pull origin lmip-biophys-stable    # sync branch

# 1. Stage inputs (symlink CRU-JRA from glade + curl Zenodo surfdata)
./scripts/data/download_lmip_data.sh --year 1920

# 2. Submit the batch job.  The script instantiates the template into a resolved
#    config.yaml under $SCRATCH, then runs the driver against it.  The env is
#    activated inside the job (override the name via LEGOESM_CONDA_ENV).
qsub -A <YOUR_ACCOUNT> scripts/cluster/derecho_lmip/lmip_biophys.pbs

# 3. Watch:
qstat -u $USER
ls -lt lmip_biophys.o*                 # PBS log lands in the submit dir
tail -f lmip_biophys.o<jobid>
```

Expected: a few minutes wall clock at 4° for the 30-day default; the console
banner prints `template:`, resolved `overrides:`, and `freeze_thaw=on`; the
experiment lands in `$SCRATCH/lmip_lmip_canopy_4deg_smoke/` with a resolved
`config.yaml`, a monthly `lmip_biophys.monthly.nc` tape, and a
`restart_<year>_d<DDD>h<HH>.npz` end-state.

**Sanity to check on the output** (why this run exists): no NaN gate trip, and a
sane surface albedo north of ~50°N — the soil freeze/thaw zero-curtain + the
per-cell canopy soil/vegetation albedo fix are exactly what this smoke validates
at scale.

## Overrides (any `qsub -v VAR=…`)

All optional — unset means "use the template's own value".

| var | meaning |
|---|---|
| `TEMPLATE` | template under `templates/land/` (default `biophysics/lmip_canopy_4deg_smoke`; e.g. `biophysics/lmip_canopy_10yr`) |
| `YEAR` / `YEAR_END` | CRU-JRA `forcing.year_start` / `year_end` (must be staged locally) |
| `RESOLUTION` | latlon N (N × 2N): 45 = 4°, 90 = 2° |
| `DT` | timestep [s] (3600 = 1 h) |
| `N_STEPS` | number of land steps (720 × 1 h = 30 days) |
| `START_DOY` | starting day-of-year (0 = Jan, deep NH winter) |
| `SURFACE_SCHEME` | `two_leaf_canopy` (default) or `simple_seb` (pair with `BULK=constant`) |
| `BULK` | `most` (default) or `constant` |
| `FREEZE_THAW` | `true` (template default) or `false` (reproduce the pre-fix NaN signature) |
| `RESTART_FROM` | warm-start seed `.npz` for a chained spin-up; empty = cold start |
| `OUT_TAG` | suffix appended to the experiment dir name |

Examples:
```bash
# 2° production 10-yr spin-up (real thing; long walltime)
qsub -A <ACCOUNT> -v TEMPLATE=biophysics/lmip_canopy_10yr scripts/cluster/derecho_lmip/lmip_biophys.pbs

# freeze/thaw OFF control (reproduce the boreal-NaN baseline for comparison)
qsub -A <ACCOUNT> -v FREEZE_THAW=false,OUT_TAG=_noft scripts/cluster/derecho_lmip/lmip_biophys.pbs

# 90-day 4° smoke starting mid-July
qsub -A <ACCOUNT> -v N_STEPS=2160,START_DOY=196,OUT_TAG=_jul90 scripts/cluster/derecho_lmip/lmip_biophys.pbs
```
