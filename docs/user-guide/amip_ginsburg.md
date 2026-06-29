# AMIP production runs on Ginsburg (Columbia)

Machine-specific runbook for the authoritative AMIP config
([`config/amip/amip_production.yaml`](../../config/amip/amip_production.yaml), PR #636/#638)
on the Columbia **Ginsburg** cluster. The authoritative YAML is machine-independent;
this branch adds the Ginsburg paths + the prep helpers that stage the inputs LOCALLY
(downloaded once, not streamed at run time).

Land comes from **ETOPO topography** (`--topography` → `load_real_topography` derives the
land fraction from sub-grid elevation>0), matching the authoritative config. No separate
land-sea mask is needed.

## 0. Environment

- `.venv/` (uv-built, GPU jax 0.10.2). The `gs://` ingest needs the `data` extra:
  `uv sync --extra dev --extra data` (or `uv pip install gcsfs fsspec` into the venv).
- Cluster: `--account=glab`, `--partition=burst` (14-day gpu) for long runs / `short` (12 h) for tests.
- Inputs live under `data/amip/` (gitignored).

## 1. Stage the inputs (prep helpers — all on this branch)

```bash
cd /burg-archive/glab/users/ac5006/legoESM
PY=.venv/bin/python ; mkdir -p data/amip

# (a) ERA5 IC — download one snapshot from the public WB2 ARCO-ERA5 to a LOCAL zarr:
$PY scripts/data/prep_era5_ic_from_zarr.py --year 1979 --month 1 --day 1 --hour 0 \
    --out data/amip/era5_ic_1979-01-01.zarr        # ~0.6 MB, 6 vars, L13   [DONE]

# (b) Synthetic CMIP6-schema forcing deck (solar/ozone/ghg/aerosol/volcanic + synthetic SST/SIC):
$PY scripts/data/generate_amip_forcing.py --out data/amip/forcing_amip \
    --start-year 1979 --end-year 2014                                       # [DONE]

# (c) Topography → land. Download a NOAA ETOPO (or GEBCO/ETOPO5) elevation NetCDF, then regrid:
#     NOAA ETOPO 2022 (public, no account): https://www.ncei.noaa.gov/products/etopo-global-relief-model
$PY scripts/data/prep_etopo_topography.py --input <etopo_source.nc> \
    --out data/amip/etopo_0p25deg.nc --resolution-deg 0.25                  # <-- TODO: get the source file
```

Status: (a) and (b) are staged. (c) needs the one real download — the NOAA ETOPO source
file — then the helper regrids it; the helper + its round-trip through `load_real_topography`
are tested (`tests/unit/test_prep_etopo_topography.py`).

## 2. Machine paths

`config/amip/amip_production.ginsburg.sh` wires the local inputs to `run_amip.py` flags
(env-overridable). Default SST/SIC = the **synthetic** deck (vars `sst`/`sic`); the YAML's
`tosbcs`/`siconcbcs` are for the ESGF override (Stage 3).

## 3. Staged runs (GPU)

```bash
source config/amip/amip_production.ginsburg.sh   # sets PY + AMIP_PATH_FLAGS

# Stage 0 — SMOKE (no real inputs): authoritative physics, flat topo, 1 day.
JAX_ENABLE_X64=1 "$PY" -u scripts/run/run_amip.py --config config/amip/amip_production.yaml \
    --ic uniform --topography flat --days 1 --diag-days 1 --output results/amip/smoke

# Stage 1 — ERA5-IC + ETOPO land + synthetic forcing (needs the ETOPO file from §1c):
JAX_ENABLE_X64=1 "$PY" -u scripts/run/run_amip.py --config config/amip/amip_production.yaml \
    "${AMIP_PATH_FLAGS[@]}" --days 10 --diag-days 5 --output results/amip/era5ic_10d

# Stage 2 — FAITHFUL (needs ESGF observed SST): swap the SST source, then a long run:
#   SST_FILE=data/amip/tosbcs_*.nc SIC_FILE=data/amip/siconcbcs_*.nc \
#   SST_VAR=tosbcs SIC_VAR=siconcbcs source config/amip/amip_production.ginsburg.sh
JAX_ENABLE_X64=1 "$PY" -u scripts/run/run_amip.py --config config/amip/amip_production.yaml \
    "${AMIP_PATH_FLAGS[@]}" --days 365 --diag-days 30 --checkpoint-days 30 \
    --output results/amip/prod_1979
```

Validate any finished run: `$PY scripts/validate/validate_amip_run.py results/amip/<dir>`.

## 4. SLURM (long runs)

Adapt `scripts/cluster/amip/submit_amip_1979_etopo_chain.sh` to a glab/`burst` self-chaining
sbatch: `cd $REPO; source config/amip/amip_production.ginsburg.sh`, call `run_amip.py` as
above, set `JAX_COMPILATION_CACHE_DIR=$OUTDIR/jax_cache`, re-`sbatch` until `TARGET_DAYS`.

## Open items (yours)

- **ETOPO source file** (§1c) — the one real download; then the helper regrids it.
- **ESGF account** — observed SST/SIC for Stage 2 (`stage_amip_realdata.py --print-esgf`).
  Stages 0–1 need none.
</content>
