# AMIP Experiment

Atmosphere-only integration with prescribed sea surface temperature (SST) and
sea-ice concentration (SIC) from observational datasets.

## Overview

The AMIP driver (`scripts/run_amip.py`) couples:

- **Dynamics**: Hydrostatic primitive equations on cubed-sphere (centered)
- **Radiation**: Selectable via `--radiation {gray,rrtmg}`
  - **Gray**: Two-stream (Frierson 2006) with moisture-dependent LW optical depth
  - **RRTMG**: RRTMGP correlated-k (Pincus et al. 2019) via bundled jax-rrtmgp
- **Convection**: Simplified Betts-Miller (SBM)
- **Boundary layer**: Bulk aerodynamic heat and moisture exchange
- **Large-scale condensation**: Saturation adjustment with latent heating
- **Friction**: Rayleigh drag (strong in BL, weak free-atmosphere)
- **Surface**: Prescribed SST + SIC from NetCDF, blending surface temperature,
  albedo, and emissivity

## Radiation modes

### Gray radiation (default)

The Frierson (2006) gray two-stream scheme:
- LW optical depth depends on latitude and moisture (tau_moist_coeff * q_v)
- SW uses Beer-Lambert absorption (no scattering)
- Seasonal solar cycle via daily-mean insolation
- Fast, stable, well-tested; suitable for idealized experiments

```bash
JAX_ENABLE_X64=1 python scripts/run_amip.py \
    --radiation gray \
    --dataset cobe --forcing-path /path/to/COBE-SST2.nc \
    --days 30 --resolution 16 --dt 600
```

### RRTMG radiation

RRTMGP correlated-k radiation (Pincus et al. 2019):
- 128 LW g-points, 112 SW g-points
- Full gas absorption: H2O (interactive), CO2, CH4, N2O, O3
- Seasonal solar geometry (daily-mean cos zenith)
- Sea-ice/ocean blended surface albedo and emissivity

```bash
JAX_ENABLE_X64=1 python scripts/run_amip.py \
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
| O3 | **Active, prescribed** | US Std Atm 1976 fit | Gaussian profile peaking near 10 hPa; not interactive |
| CFCs | Not included | — | Not in current gas optics files |
| Clouds | **Clear-sky only** | — | Cloud-radiation interaction not yet wired |
| Aerosols | **Not included** | — | Clear-sky; no aerosol optical depth |

#### RRTMG scientific limitations

1. **Clear-sky radiation**: No cloud-radiation interaction. The infrastructure
   exists (cloud optics lookup tables are bundled, `include_clouds` config flag
   exists, microphysics schemes produce q_c/q_i) but the coupling is not yet
   wired. This means the model overestimates surface SW and underestimates
   planetary albedo relative to observations.

2. **Scalar surface albedo**: The bundled jax-rrtmgp uses a single scalar
   `sfc_alb` for all columns. The AMIP driver passes the column-mean of the
   ice/ocean blended albedo. At coarse resolution (C16-C48) this is a good
   approximation; at fine resolution with large ice fraction gradients it may
   introduce small errors.

3. **Prescribed ozone**: The O3 profile is a fixed analytical fit to the US
   Standard Atmosphere 1976. It does not vary with latitude, season, or
   chemistry. A proper ozone climatology (e.g., from CMIP6 forcing files)
   would improve stratospheric heating.

4. **No aerosols**: Aerosol direct and indirect effects are absent.

5. **Uniform well-mixed gases**: CO2, CH4, N2O are spatially and temporally
   uniform. Time-varying concentrations from CMIP forcing files are scaffolded
   (`ExternalForcingConfig`) but not yet implemented.

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
JAX_ENABLE_X64=1 python scripts/run_amip.py \
    --dataset cobe \
    --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \
    --days 30 --resolution 16 --dt 600
```

### RRTMG radiation with checkpointing

```bash
JAX_ENABLE_X64=1 python scripts/run_amip.py \
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
JAX_ENABLE_X64=1 python scripts/run_amip.py \
    --radiation rrtmg \
    --dataset cobe \
    --forcing-path /path/to/COBE-SST2.nc \
    --co2-ppmv 830 \
    --days 200
```

### Restart from checkpoint

```bash
JAX_ENABLE_X64=1 python scripts/run_amip.py \
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

### Component status

| Component | Status | Notes |
|-----------|--------|-------|
| SST/SIC forcing | **Active** | COBE-SST2, HadISST presets; custom supported |
| Gray radiation | **Active** | Frierson 2006, moist LW OD, seasonal solar |
| RRTMG radiation | **Active** | Correlated-k, H2O+CO2+CH4+N2O+O3, clear-sky |
| SBM convection | **Active** | Frierson 2007 |
| BL exchange | **Active** | Bulk aerodynamic (heat + moisture) |
| Large-scale condensation | **Active** | Saturation adjustment |
| Rayleigh friction | **Active** | BL + free-atmosphere drag |
| Checkpoint/restart | **Active** | NPZ-based, reproducible |
| Experiment config | **Active** | JSON-serializable `AMIPExperimentConfig` |
| Cloud-radiation coupling | **Scaffolded** | Cloud optics exist but not wired to RRTMG |
| GHG time-varying | **Scaffolded** | `ExternalForcingConfig` interface ready |
| Ozone climatology | **Placeholder** | Fixed US Std Atm profile; no lat/season dependence |
| Aerosol forcing | **Placeholder** | Config exists; not connected to radiation |
| Solar TSI variation | **Scaffolded** | Constant TSI active; time-varying from file not yet |

## Recommended stable settings

For a first run:

- **Resolution**: C16/L20 (fast, stable)
- **Time step**: 600 s (safe for C16; use 450 s for C24, 300 s for C48)
- **Duration**: 30 days for testing, 200+ days for spinup
- **Diagnostics**: every 5 days (`--diag-days 5`)
- **Radiation**: `gray` for fast iteration, `rrtmg` for realistic forcing
- **Radiation cadence**: every step for gray; `--rad-update-steps 6` for RRTMG at C16

The hydrostatic centered discretization is the most stable and well-tested
path. Do not use the FV cubed-sphere path for AMIP unless explicitly validated.

## CLI reference

```
--dataset {cobe,hadisst,custom}   Forcing dataset preset
--forcing-path PATH               Path to NetCDF forcing file (required unless restarting)
--sst-var NAME                    SST variable name (custom only)
--sic-var NAME                    SIC variable name (custom only)
--sst-offset FLOAT                Additive offset for SST (e.g., 273.15)
--sic-scale FLOAT                 Multiplicative scale for SIC (e.g., 0.01)
--start-day FLOAT                 Start day within forcing record
--days INT                        Integration length [days]
--resolution INT                  Cubed-sphere N
--nlev INT                        Number of vertical levels
--dt FLOAT                        Time step [seconds]
--diag-days INT                   Diagnostic output interval [days]
--output PATH                     Output directory
--checkpoint-days INT             Checkpoint interval [days]; 0 = off (default)
--restart-from PATH               Restart from checkpoint .npz file
--radiation {gray,rrtmg}          Radiation scheme (default: gray)
--rad-update-steps INT            Radiation call frequency [steps] (default: 1)
--co2-ppmv FLOAT                  CO2 concentration [ppmv] for RRTMG (default: 415)
--ch4-ppbv FLOAT                  CH4 concentration [ppbv] for RRTMG (default: 1900)
--n2o-ppbv FLOAT                  N2O concentration [ppbv] for RRTMG (default: 332)
```
