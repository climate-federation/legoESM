# LMIP biophysics on Derecho — quick runbook

The M3 smoke run for the biophysics-only LMIP driver (branch
`land-dev-followup`).  Runs `scripts/run/run_lmip_biophys.py` against the real
CRU-JRA reanalysis on glade + the CLM5 surfdata hosted on Zenodo.

## One-time setup

```bash
# 1. clone + branch
cd /glade/work/$USER          # or wherever you keep code
git clone https://github.com/climate-federation/legoESM.git
cd legoESM
git checkout land-dev-followup

# 2. Python env (see docs/getting_started.md; the federation install script
#    handles the multi-package workspace under plain pip)
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
git pull origin land-dev-followup      # sync branch (add cf remote if origin is a fork)

# 1. Stage inputs (symlink CRU-JRA from glade + curl Zenodo surfdata)
./scripts/data/download_lmip_data.sh --year 1920

# 2. Submit the batch job (the script activates the env itself; override the
#    env name via LEGOESM_CONDA_ENV if yours is not named 'legoesm-lmip').
qsub -A <YOUR_ACCOUNT> scripts/cluster/derecho_lmip/lmip_biophys.pbs
qsub -A <YOUR_ACCOUNT> -v LEGOESM_CONDA_ENV=my-env scripts/cluster/derecho_lmip/lmip_biophys.pbs

# 3. Watch:
qstat -u $USER
ls -lt lmip_biophys.o*                 # PBS log lands in the submit dir
tail -f lmip_biophys.o<jobid>
```

Expected: ~10-15 min wall clock at 2° for 10 model days; console line ending
`-> PASS`; a `lmip_biophys.nc` in `$SCRATCH/lmip_biophys_1920/`.

## Overrides (any `qsub -v VAR=…`)

| var | default | meaning |
|---|---|---|
| `YEAR` | 1920 | CRU-JRA forcing year (must be staged locally) |
| `RESOLUTION` | 90 | latlon N (grid is N × 2N) — 90 = 2° |
| `DT` | 3600 | timestep in seconds (1 h; `1800` for 30-min) |
| `N_STEPS` | 240 | number of land steps (240 × 1 h = 10 days) |
| `START_DOY` | 196 | starting day-of-year |
| `SURFACE_SCHEME` | `two_leaf_canopy` | or `simple_seb` (faster, no canopy) |
| `OUT_TAG` | *(empty)* | suffix appended to `$SCRATCH/lmip_biophys_${YEAR}${OUT_TAG}` |

Example — faster smoke on the simple SEB, 3 model days:
```bash
qsub -A <ACCOUNT> -v N_STEPS=72,SURFACE_SCHEME=simple_seb,OUT_TAG=_seb_smoke \
    scripts/cluster/derecho_lmip/lmip_biophys.pbs
```
