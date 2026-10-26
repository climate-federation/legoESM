# AMIP Experiment

Atmosphere-only integration with prescribed sea surface temperature (SST) and
sea-ice concentration (SIC) from observational datasets.

## Production configuration (legoESM 1.0)

The authoritative production AMIP configuration is the **CAM6 physics suite on
the MPAS/Voronoi grid**, single-sourced in `config/amip/amip_production.yaml`
(machine paths and a ready launcher in `config/amip/amip_production.sh`):

```bash
PY=.venv/bin/python DAYS=60 OUTDIR=results/amip_prod   # your choices
source config/amip/amip_production.sh                  # sets AMIP_PATH_FLAGS
"${PY}" -u scripts/run/run_amip.py \
    --config config/amip/amip_production.yaml \
    "${AMIP_PATH_FLAGS[@]}" --days "${DAYS}" --output "${OUTDIR}"
```

What the YAML selects (read the file for every value; it is the record):

- **Grid / dycore**: MPAS Voronoi level 6 (40,962 cells, ~1.1°), CAM 32-level
  hybrid table (`vertical_coord: cam_l32`), dt 112.5 s, fp64.
- **Physics step**: every 1800 s, with three macro/micro sub-steps; radiation hourly.
- **Radiation**: RRTMG with McICA cloud overlap, no in-cloud inhomogeneity thinning.
- **Convection**: Zhang–McFarlane (CAM6 port).
- **Turbulence + cloud fraction**: prognostic CLUBB; `clouds: cam6_clubb`.
- **Microphysics**: Morrison double-moment with CAM6 MG2 in-cloud warm rain.
- **Gravity-wave drag**: McFarlane (orographic).
- **Surface**: CESM Large–Yeager bulk fluxes; interactive multilayer land.

Production arms run on 4+ GPUs. The previous production deck (Sundqvist
clouds, L36) is preserved runnable as `config/amip/amip_sundqvist_l36.yaml`.

## How to run an AMIP CMIP simulation

Production AMIP is `scripts/run/run_amip.py --config
config/amip/amip_production.yaml` plus the machine input paths of
`config/amip/amip_production.sh` (Levante) or
`config/amip/amip_production.ginsburg.sh` (Ginsburg).

`scripts/run/run_amip_smoke_deck.py` (formerly `run_amip_cmip6_deck.py`) is
NOT production: it wraps `run_amip.py` with a historical C16 Sundqvist / SBM /
Louis smoke stack on flat terrain (no land) and synthetic forcing by default,
to exercise the transient forcing plumbing (prescribed SST/SIC + GHG, ozone,
solar, aerosol, volcanic) on **every grid** (cubed-sphere, lat-lon,
Gaussian/spectral, MPAS/Voronoi).

Pick the level that matches what you want:

### A. Quickest — pipeline smoke (synthetic forcing, minutes)

Confirms the model runs end-to-end on every grid.  No data needed; the
deck auto-generates synthetic CMIP6-shape forcing.

```bash
# All 8 (grid, discretization) cases, gray radiation, 1-day each:
python scripts/validate/smoke_test_amip_all_grids.py --days 1
#   cubed_sphere/{centered,finite_volume,cdgrid}
#   latlon/{centered,finite_volume,latlon_cgrid}
#   gaussian/spectral
#   voronoi/mpas   (standard dt; integrator auto-mapped to ssp_rk54_scan)

# Or a single short run on one grid:
JAX_ENABLE_X64=1 python scripts/run/run_amip_smoke_deck.py \
    --grid-type cubed_sphere --discretization finite_volume \
    --resolution 16 --days 30 --ic default \
    --output results/amip_smoke
```

> The synthetic deck has **fake SST** (noise) — fine for a pipeline check,
> NOT scientific AMIP.  For realism use the real-data path below.

### B. Realistic — ERA5 initial condition (no credentials)

The deck default `--ic era5` pulls the initial atmospheric state from the
**public ARCO ERA5** store on GCS (no account needed), giving real winds +
moisture and avoiding the cold-start drift of the uniform IC.  ERA5 IC is
wired on cubed-sphere / lat-lon / Gaussian (MPAS falls back to the uniform
IC for now).

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip_smoke_deck.py \
    --grid-type cubed_sphere --discretization finite_volume \
    --resolution 36 --days 120 --dt-auto \
    --radiation rrtmg --ic era5 \
    --output results/amip_era5
#   --ic era5 with no --ic-path -> public ARCO ERA5 automatically.
```

### C. Faithful CMIP — real observed SST + ERA5 IC (full combination)

A scientifically-faithful AMIP run needs the four pieces together:
**correct GHG** (built-in) + **ERA5 IC** (public ARCO) + **real observed
SST** (PCMDI / input4MIPs AMIP II bcs — needs a free ESGF account to
download) + **months of spin-up** (checkpointed).

```bash
# 1. Stage + validate (prints the ESGF download steps and a ready command):
python scripts/data/stage_amip_realdata.py --print-esgf
python scripts/data/stage_amip_realdata.py --sst-file /data/tosbcs_input4MIPs_*.nc

# 2. One 30-year run with real SST + ERA5 IC:
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 python scripts/run/run_amip_smoke_deck.py \
    --grid-type cubed_sphere --discretization finite_volume \
    --resolution 36 --days 10950 --dt-auto \
    --rad-update-steps 18 --diag-days 30 --checkpoint-days 365 \
    --radiation rrtmg --ic era5 \
    --sst-file /data/tosbcs_input4MIPs_*_PCMDI-AMIP-1-1-9_gn_*.nc \
    --output results/amip30y_cube

# 3. Or all four grids, checkpointed + resumable, via the launcher:
export ERA5_IC_PATH=gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
export SST_FILE=/data/tosbcs_input4MIPs_*_PCMDI-AMIP-1-1-9_gn_*.nc
bash scripts/run/run_amip30y_allgrids_local.sh
```

The deck defaults the real-SST var/unit flags to the input4MIPs convention
(`tosbcs` in K, `siconcbcs` in %); the loader's units-attribute guard
rejects a wrong file/flag combination loudly.

### Deck physics defaults

The deck's defaults (override any with the matching flag; this is NOT the
CAM6 production configuration above): **RRTMG correlated-k** radiation,
**Morrison double-moment** microphysics, **Sundqvist** cloud fraction,
**SBM** convection, **Louis** PBL, **McFarlane (orographic)** GWD, with
**aerosol→CCN** coupling (Andreae 2009) and **zenith-dependent ocean albedo**
(Briegleb 1992) on the pipeline grids.
`--dt-auto` picks each grid's stability-ladder timestep (C36→150 s,
latlon72→75 s, T47→150 s, voronoi→300 s) so long runs cannot blow up.

### 2.5° resolution per grid

| Grid | `--grid-type / --discretization` | `--resolution` |
|------|----------------------------------|----------------|
| Cubed-sphere | `cubed_sphere / finite_volume` | `36` (C36) |
| Lat-lon | `latlon / finite_volume` | `72` |
| Gaussian | `gaussian / spectral` | `47` (T47) |
| Voronoi/MPAS | `voronoi / mpas` | `5` (level 5) |

### Forcing files

`forcing_amip/` holds the six CMIP6-schema files (auto-generated synthetic
when missing, or drop real input4MIPs files in with these names):

| File | Purpose | Schema reference |
|------|---------|------------------|
| `sst_sic_amip_<sy>-<ey>.nc` | SST + SIC monthly | HadISST (or `--sst-file` for input4MIPs AMIP II bcs) |
| `ghg_amip_<sy>-<ey>.nc` | Annual GHG | input4MIPs `greenhouse_historical_plus` |
| `ozone_amip_clim.nc` (or `_<sy>-<ey>.nc`) | Ozone clim or interannual | input4MIPs vmro3 (CCMI-1-0) |
| `solar_amip_<sy>-<ey>.nc` | Daily TSI + 14-band | MPI-M `swflux_14band_cmip6_*` |
| `aerosol_amip_clim.nc` | Monthly zonal AOD | Kinne |
| `volcanic_amip_<sy>-<ey>.nc` | Volcanic AOD | CMIP6 `bc_aeropt_cmip6_volc_*` |

Regenerate / seed the synthetic deck (non-SST channels are units-correct):

```bash
python scripts/data/generate_amip_forcing.py --out forcing_amip \
    --start-year 1979 --end-year 2014
```

### Validate a finished run

```bash
python scripts/validate/validate_amip_run.py results/amip30y_cube
```

See [`amip_cmip_allgrids.md`](amip_cmip_allgrids.md) for the all-grids
forcing/physics-wiring runbook and [`forcing/AMIP.md`](forcing/AMIP.md) for
the build trace.

## Overview

The AMIP driver (`scripts/run/run_amip.py`) couples the components below.
Defaults shown are the bare-CLI defaults; production runs set every scheme
explicitly through `--config` (see above).

- **Dynamics**: `--grid-type {cubed_sphere,gaussian,latlon,mpas}` (default
  `cubed_sphere`) with `--discretization`; `--vertical-coord {sigma,hybrid,cam_l32}`
- **Radiation**: `--radiation {none,gray,rrtmgp,rrtmg}` (default `gray`)
  - **Gray**: Two-stream (Frierson 2006) with moisture-dependent LW optical depth
  - **RRTMGP** (`rrtmgp`; `rrtmg` is an alias for the same scheme): correlated-k
    (Pincus et al. 2019) via bundled jax-rrtmgp
  - Optional diurnal cycle (`--diurnal-cycle`) with instantaneous solar zenith angle
- **Convection**: `--convection` (default `tiedtke`; production `zhang_mcfarlane`)
- **Turbulence**: `--turbulence` (default `louis`; production `clubb`)
- **Surface fluxes**: `--surface-bulk-scheme {constant,coare3,large_yeager,large_yeager_cesm}` (default `constant`)
- **Clouds**: `--clouds {none,sundqvist,xu_randall,cam6_clubb}` (default `xu_randall`)
- **Microphysics**: `--microphysics {none,kessler,sundqvist,seifert_beheng,morrison,thompson,p3,sdm,fast_sbm}` (default `sundqvist`)
- **Gravity-wave drag**: `--gravity-wave-drag` (default `mcfarlane`; `+`-joined compositions such as `hines+mcfarlane` accepted)
- **Ozone**: `--ozone-source {standard,analytical,mls,none}` or `--ozone-forcing external --ozone-file …`
- **GHG / aerosol**: `--ghg-forcing {constant,external}`, `--aerosol-forcing {off,external}`, `--aerosol-ccn`
- **Land**: `--use-multilayer-land` (interactive Richards land; production on)
- **Surface**: Prescribed SST + SIC from NetCDF, blending surface temperature,
  albedo, and emissivity
- **Diagnostics**: Energy budget tracking, optional monthly-mean accumulation (`--monthly-means`)

## Radiation modes

### Gray radiation (default)

The Frierson (2006) gray two-stream scheme:
- LW optical depth depends on latitude and moisture: dtau_k = tau_moist_coeff * q_v * dp_k / g [m²/kg]
- SW uses Beer-Lambert absorption (no scattering)
- Seasonal solar cycle; daily-mean insolation (default) or diurnal cycle
- Fast, stable, well-tested; suitable for idealized experiments

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --radiation gray \
    --dataset cobe --forcing-path /path/to/COBE-SST2.nc \
    --days 30 --resolution 16 --dt 600
```

### RRTMG radiation

RRTMGP correlated-k radiation (Pincus et al. 2019):
- 128 LW g-points, 112 SW g-points
- Full gas absorption: H2O (interactive), CO2, CH4, N2O, O3
- Seasonal solar geometry; daily-mean (default) or diurnal cycle cos zenith
- Sea-ice/ocean blended surface albedo and emissivity
- Optional cloud-radiation coupling (Sundqvist or Xu-Randall cloud fraction)
- Analytical ozone profile with latitude dependence

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --radiation rrtmg \
    --dataset cobe --forcing-path /path/to/COBE-SST2.nc \
    --days 30 --resolution 16 --dt 600 \
    --co2-ppmv 415 --ch4-ppbv 1900 --n2o-ppbv 332
```

#### RRTMG atmospheric composition

| Species | Status | Source | Notes |
|---------|--------|--------|-------|
| H2O | **Active, interactive** | Prognostic q_v | Radiatively coupled to dynamics |
| CO2 | **Active, prescribed** | CLI `--co2-ppmv` | Default 415 ppmv; uniform in space and time |
| CH4 | **Active, prescribed** | CLI `--ch4-ppbv` | Default 1900 ppbv; uniform |
| N2O | **Active, prescribed** | CLI `--n2o-ppbv` | Default 332 ppbv; uniform |
| O3 | **Active, prescribed** | `--ozone-source` | `standard` (US Std Atm), `analytical` (lat-dependent Gaussian), `mls` (SAM RCEMIP MLS climatology), or `none` |
| CFCs | Not included | — | Not in current gas optics files |
| Clouds | **Active (optional)** | `--clouds` | `none` (clear-sky), `sundqvist`, `xu_randall`, or `cam6_clubb`; coupled to RRTMG cloud optics |
| Aerosols | **Active (optional)** | `--aerosol-forcing external` | Kinne-style AOD with per-shortwave-band SSA/asymmetry; volcanic via `--volcanic-aerosol-file` |

#### RRTMG scientific limitations

1. **Inline ozone by default**: without `--ozone-forcing external` the O3
   profile is a fixed analytical/standard profile with no seasonal cycle or
   chemistry. Pass a CMIP6 `vmro3` file for a real climatology.

2. **Aerosols off by default**: aerosol direct and CCN effects need
   `--aerosol-forcing external` (and `--aerosol-ccn`).

3. **Uniform well-mixed gases**: CO2, CH4, N2O are spatially uniform. They
   are constant (CLI flags) unless `--ghg-forcing external --ghg-file …`
   supplies a time series.

#### Radiation update interval

For performance, RRTMG can be called less frequently than every dynamics step:

```bash
--rad-update-steps 6   # recompute radiation every 6 steps (= 1 hour at dt=600s)
```

Between radiation calls, the heating rate tendencies are held constant. This
is standard practice in GCMs (CESM uses 1-hour radiation cadence). For gray
radiation, every-step calling is fast enough that sub-cycling is unnecessary.

## Required forcing file structure

The NetCDF file must contain:

| Variable | Description | Units |
|----------|-------------|-------|
| SST      | Sea surface temperature | K or C (see presets) |
| SIC      | Sea-ice concentration | fraction or % (see presets) |
| lat      | Latitude | degrees N |
| lon      | Longitude | degrees E |
| time     | Time coordinate | datetime64 or numeric days |

Variable names depend on the dataset preset.

## Supported presets

| Preset | SST var | SIC var | SST units | SIC units | Source |
|--------|---------|---------|-----------|-----------|--------|
| `cobe` | `SST_cpl` | `ice_cov` | K | % | COBE-SST2 (PCMDI) |
| `hadisst` | `sst` | `sic` | C | fraction | Met Office HadISST |
| `custom` | user-specified | user-specified | user-specified | user-specified | Any NetCDF |

## Example commands

### Gray radiation (default, fast)

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --dataset cobe \
    --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \
    --days 30 --resolution 16 --dt 600
```

### RRTMG radiation with checkpointing

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --radiation rrtmg \
    --dataset hadisst \
    --forcing-path /path/to/HadISST_sst.nc \
    --days 365 --resolution 24 --dt 450 \
    --rad-update-steps 6 \
    --checkpoint-days 30 \
    --output results/amip_rrtmg_1yr
```

### RRTMG with doubled CO2

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --radiation rrtmg \
    --dataset cobe \
    --forcing-path /path/to/COBE-SST2.nc \
    --co2-ppmv 830 \
    --days 200
```

### Full-physics RRTMG with clouds and diurnal cycle

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --radiation rrtmg \
    --dataset cobe \
    --forcing-path /path/to/COBE-SST2.nc \
    --days 365 --resolution 48 --nlev 40 --dt 450 \
    --diurnal-cycle \
    --ozone-source analytical \
    --clouds xu_randall \
    --microphysics kessler \
    --monthly-means \
    --rad-update-steps 3 \
    --checkpoint-days 30 \
    --output results/amip_full_physics
```

### Long C48 AMIP (not the production configuration)

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --radiation rrtmg \
    --dataset cobe --forcing-path /path/to/COBE-SST2.nc \
    --days 3650 --resolution 48 --nlev 40 --dt 450 \
    --diurnal-cycle \
    --ozone-source analytical \
    --clouds xu_randall \
    --microphysics kessler \
    --monthly-means \
    --rad-update-steps 6 \
    --checkpoint-days 30 \
    --output results/amip_production_10yr
```

This runs C48/L40 with RRTMG, diurnal cycle, analytical ozone, Xu-Randall
clouds, Kessler microphysics, and monthly-mean diagnostics.

### Restart from checkpoint

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip.py \
    --restart-from results/amip_rrtmg_1yr/checkpoint_day_0030.npz \
    --forcing-path /path/to/HadISST_sst.nc \
    --days 365
```

## Expected outputs

Each run creates a timestamped subdirectory under `results/amip/` (or the
path given by `--output`) containing:

| File | Description |
|------|-------------|
| `experiment_config.json` | Serialized `AMIPExperimentConfig` for reproducibility |
| `results.txt` | Configuration, radiation scheme, diagnostics table, wall time |
| `timeseries.npz` | NumPy archive with all time series arrays |
| `amip_timeseries.png` | 8-panel time series (SST, SIC, T, wind, precip, CWV, TOA fluxes, p_s) |
| `amip_final_state.png` | Global lat-lon maps (SST, SIC, T_low, q_v, precip, TOA net, sfc net, wind) |
| `amip_snapshots.png` | Lat-lon snapshot evolution with radiative fields |
| `amip_profiles.png` | Vertical profile evolution (T, q_v) |
| `checkpoint_day_NNNN.npz` | Checkpoint files (if `--checkpoint-days` > 0) |
| `checkpoint_final.npz` | Final checkpoint (if checkpointing enabled) |
| `monthly_means.npz` | Zonal/global monthly means (if `--monthly-means` enabled) |

## Diagnostics

### Active diagnostics

| Diagnostic | Description | Available in |
|------------|-------------|--------------|
| Global-mean SST, SIC | Prescribed boundary conditions | Both |
| Global-mean T_atm, T_low | Atmospheric temperature | Both |
| Max wind speed | Stability monitor | Both |
| Precipitation | Convective + large-scale [mm/day] | Both |
| CWV | Column water vapor [kg/m2] | Both |
| TOA SW up | Reflected shortwave at TOA [W/m2] | Both |
| TOA LW up | Outgoing longwave at TOA [W/m2] | Both |
| Surface SW net | Net shortwave at surface [W/m2] | Both |
| Surface LW net | Net longwave at surface [W/m2] | Both |
| Mean surface pressure | Dry-mass conservation proxy [Pa] | Both |
| Vertical profiles | T(sigma) and q_v(sigma) | Both |
| 2D snapshots | Maps at selected days | Both |
| TOA net radiation | R_TOA = SW↓ - SW↑ - LW↑ [W/m²] | Both |
| Column energy | Moist static energy [J/m²] | Both |
| Energy tendency dE/dt | Column energy change rate [W/m²] | Both |
| Energy residual | R_TOA − dE/dt [W/m²] | Both |
| Monthly zonal means | Latitude-binned monthly averages | `--monthly-means` |
| Monthly profiles | Zonal-mean T, u, q_v profiles | `--monthly-means` |

### Component status

| Component | Status | Notes |
|-----------|--------|-------|
| SST/SIC forcing | **Active** | COBE-SST2, HadISST presets; custom supported |
| Gray radiation | **Active** | Frierson 2006, moist LW OD, seasonal solar |
| RRTMG radiation | **Active** | Correlated-k, H2O+CO2+CH4+N2O+O3, all-sky with cloud optics |
| Convection | **Active** | `--convection` (Tiedtke default; ZM, Bechtold, SBM, KF, Emanuel, …) |
| BL exchange | **Active** | Bulk aerodynamic (constant default); COARE3/LY04/CESM-LY via `--surface-bulk-scheme` |
| Large-scale condensation | **Active** | Saturation adjustment |
| Rayleigh friction | **Active** | BL + free-atmosphere drag |
| Checkpoint/restart | **Active** | NPZ-based, reproducible |
| Experiment config | **Active** | JSON-serializable `AMIPExperimentConfig` |
| Diurnal cycle | **Active** | Instantaneous cos(SZA) per column; `--diurnal-cycle` |
| Cloud-radiation coupling | **Active** | Sundqvist or Xu-Randall cloud fraction → RRTMG optics |
| Microphysics | **Active** | Sundqvist default; Kessler, Morrison (MG2/SAM), Thompson, P3, … via `--microphysics` |
| Ozone | **Active** | Standard (US Std Atm), analytical (lat-dependent), or MLS climatology; `--ozone-source` |
| Dynamic albedo | **Active** | Temperature/zenith-dependent ice+snow albedo; `--dynamic-albedo` |
| Energy budget | **Active** | Online column energy, TOA balance, residual tracking |
| Monthly means | **Active** | Zonal-mean and global-mean monthly accumulation; `--monthly-means` |
| GHG time-varying | **Active** | `GHGConfig(source="file")` with NetCDF time interpolation |
| Experiment templates | **Active** | piControl, historical, SSP2-4.5, SSP5-8.5, AMIP, 1pctCO₂; `create_experiment_config()` |
| CMOR output | **Active** | CF-1.8/CMIP6 DRS NetCDF via `CFWriter`; 27 CMOR variables |
| Restart/reproducibility | **Active** | SHA-256 state digests, config hashes; `verify_reproducibility()` |
| Tuning validation | **Active** | `validate_tuning()` checks CFL, ranges, conflicts |
| Ozone from file | **Active** | `--ozone-forcing external --ozone-file …` (CMIP6 `vmro3`, climatology or interannual) |
| Aerosol from file | **Active** | `--aerosol-forcing external --aerosol-file …`; CCN coupling via `--aerosol-ccn` |
| Solar TSI variation | **Active** | `SolarConfig(source="file")` with NetCDF time interpolation |

## Recommended stable settings

For a first run:

- **Resolution**: C16/L20 (fast, stable)
- **Time step**: 600 s (safe for C16; use 450 s for C24, 300 s for C48)
- **Duration**: 30 days for testing, 200+ days for spinup
- **Diagnostics**: every 5 days (`--diag-days 5`)
- **Radiation**: `gray` for fast iteration, `rrtmg` for realistic forcing
- **Radiation cadence**: every step for gray; `--rad-update-steps 6` for RRTMG at C16

The hydrostatic centered discretization is the most stable and well-tested
path. The FV cubed-sphere path has been validated for 365-day integrations at
C16/L20 (LW_TOA ≈ 236 W/m², precip ≈ 4 mm/day). FC-Gram and C-grid variants
are available but not yet validated for AMIP-length runs.

## CLI reference

```
# Grid and integration
--config PATH                     YAML run config (e.g. config/amip/amip_production.yaml)
--grid-type {cubed_sphere,gaussian,latlon,mpas}  Grid (default: cubed_sphere)
--resolution INT                  Grid resolution (default: 16)
--nlev INT                        Number of vertical levels (default: 40)
--dt FLOAT                        Time step [seconds] (default: 600)
--days INT                        Integration length [days] (default: 200)
--start-day FLOAT                 Start day within forcing record
--diag-days FLOAT                 Diagnostic output interval [days] (default: 5)

# Forcing
--dataset {cobe,hadisst,custom,analytical}  Forcing dataset preset (default: analytical)
--forcing-path PATH               Path to NetCDF forcing file (required for cobe/hadisst/custom)
--sst-var NAME                    SST variable name (custom only)
--sic-var NAME                    SIC variable name (custom only)
--sst-offset FLOAT                Additive offset for SST (e.g., 273.15)
--sic-scale FLOAT                 Multiplicative scale for SIC (e.g., 0.01)

# Radiation
--radiation {none,gray,rrtmgp,rrtmg}  Radiation scheme (default: gray)
--rad-update-steps INT            Radiation call frequency [steps] (default: 1)
--diurnal-cycle                   Use instantaneous solar zenith angle (default: off)
--co2-ppmv FLOAT                  CO2 concentration [ppmv] for RRTMG (default: 415)
--ch4-ppbv FLOAT                  CH4 concentration [ppbv] for RRTMG (default: 1900)
--n2o-ppbv FLOAT                  N2O concentration [ppbv] for RRTMG (default: 332)
--ozone-source {standard,analytical,mls,none}  Ozone profile (default: standard)
--clouds {none,sundqvist,xu_randall,cam6_clubb}  Cloud fraction scheme (default: xu_randall)

# Physics
--microphysics {none,kessler,sundqvist,seifert_beheng,morrison,thompson,p3,sdm,fast_sbm}
                                           Microphysics (run_amip default: sundqvist; deck default: morrison)
--convection NAME                 Convection scheme (default: tiedtke)
--turbulence NAME                 Turbulence scheme (default: louis)
--gravity-wave-drag SPEC          GWD scheme or "+"-joined composition (default: mcfarlane)
--dynamic-albedo                  Temperature/zenith-dependent surface albedo (default: off)

# Diagnostics
--monthly-means                   Accumulate zonal/global monthly means (default: off)

# I/O
--output PATH                     Output directory
--checkpoint-days INT             Checkpoint interval [days]; 0 = off (default)
--restart-from PATH               Restart from checkpoint .npz file
```

## Post-run validation

The `scripts/validate/validate_amip_run.py` script checks run output against observational
targets:

```bash
python scripts/validate/validate_amip_run.py results/amip_run/
```

### Validation targets

| Metric | Target range | Observed | Unit |
|--------|-------------|----------|------|
| Global mean T_2m | 287–289 | ~288 | K |
| Global mean precipitation | 2.5–3.0 | ~2.7 | mm/day |
| Net TOA imbalance | < 1 | ~0.5 | W/m² |
| OLR | 235–245 | ~240 | W/m² |

The script also checks for a subtropical jet at ~30° lat, reasonable
equatorial/polar temperatures, and ITCZ position.
