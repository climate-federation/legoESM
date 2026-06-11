# AMIP Experiment

Atmosphere-only integration with prescribed sea surface temperature (SST) and
sea-ice concentration (SIC) from observational datasets.

## Quick start: CMIP6 AMIP deck

For a complete CMIP6-protocol AMIP run (transient GHG, ozone, solar, aerosol,
volcanic), use `scripts/run/run_amip_cmip6_deck.py` instead of `run_amip.py`
directly.  The deck driver auto-generates synthetic CMIP6-shape forcing files
(or consumes real ones if dropped under `forcing_amip/` with the canonical
names) and pins the canonical RRTMG + Sundqvist + Kessler + SBM + Louis stack:

```bash
JAX_ENABLE_X64=1 python scripts/run/run_amip_cmip6_deck.py \
    --resolution 16 --days 30 --output results/amip_deck_test
```

The `forcing_amip/` directory expects six files following the CMIP6 schemas:

| File                              | Purpose             | Schema reference                               |
|-----------------------------------|---------------------|------------------------------------------------|
| `sst_sic_amip_<sy>-<ey>.nc`       | SST + SIC monthly   | HadISST                                        |
| `ghg_amip_<sy>-<ey>.nc`           | Annual GHG          | input4MIPs `greenhouse_historical_plus.nc`    |
| `ozone_amip_clim.nc` (or `_<sy>-<ey>.nc`) | Ozone clim or interannual | input4MIPs vmro3 (CCMI-1-0)            |
| `solar_amip_<sy>-<ey>.nc`         | Daily TSI + 14-band | MPI-M `swflux_14band_cmip6_*`                 |
| `aerosol_amip_clim.nc`            | Monthly zonal AOD   | Kinne                                          |
| `volcanic_amip_<sy>-<ey>.nc`      | Volcanic AOD        | CMIP6 `bc_aeropt_cmip6_volc_*`                |

To regenerate (or seed) the synthetic deck:

```bash
python scripts/data/generate_amip_forcing.py --out forcing_amip \
    --start-year 1979 --end-year 2014
# Optional: interannually-varying ozone (exercises the loader's
# non-cyclic dispatch instead of the 12-month climatology):
python scripts/data/generate_amip_forcing.py --out forcing_amip \
    --start-year 1979 --end-year 2014 \
    --component ozone --ozone-interannual
```

To validate a finished run:

```bash
python scripts/validate/validate_amip_run.py results/amip_deck_test
```

To run the deck across every supported (grid, discretization)
combination as a smoke test:

```bash
python scripts/validate/smoke_test_amip_all_grids.py --days 1
# All 8 cases pass on legoESM main:
#   cubed_sphere/{centered,finite_volume,cdgrid}
#   latlon/{centered,finite_volume,latlon_cgrid}
#   gaussian/spectral
#   voronoi/mpas (standard dt; integrator auto-mapped to ssp_rk54_scan)
```

See [`forcing/AMIP.md`](forcing/AMIP.md) for the iteration-by-iteration trace
of how this infrastructure was built.

## Overview

The AMIP driver (`scripts/run/run_amip.py`) couples:

- **Dynamics**: Hydrostatic primitive equations on cubed-sphere (centered)
- **Radiation**: Selectable via `--radiation {gray,rrtmg}`
  - **Gray**: Two-stream (Frierson 2006) with moisture-dependent LW optical depth
  - **RRTMG**: RRTMGP correlated-k (Pincus et al. 2019) via bundled jax-rrtmgp
  - Optional diurnal cycle (`--diurnal-cycle`) with instantaneous solar zenith angle
- **Convection**: Simplified Betts-Miller (SBM)
- **Boundary layer**: Bulk aerodynamic heat and moisture exchange (constant coefficients default; MOST/COARE3/LY04 available via coupler `bulk_scheme`)
- **Large-scale condensation**: Saturation adjustment with latent heating
- **Clouds**: Diagnostic cloud fraction (`--clouds {none,sundqvist,xu_randall}`) coupled to RRTMG radiation
- **Microphysics**: Selectable via `--microphysics {none,kessler,sundqvist}`
- **Ozone**: Selectable via `--ozone-source {standard,analytical,none}`
- **Friction**: Rayleigh drag (strong in BL, weak free-atmosphere)
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
| O3 | **Active, prescribed** | `--ozone-source` | `standard` (US Std Atm), `analytical` (lat-dependent Gaussian), or `none` |
| CFCs | Not included | — | Not in current gas optics files |
| Clouds | **Active (optional)** | `--clouds` | `none` (clear-sky), `sundqvist`, or `xu_randall`; coupled to RRTMG cloud optics |
| Aerosols | **Not included** | — | Clear-sky; no aerosol optical depth |

#### RRTMG scientific limitations

1. **Scalar surface albedo**: The bundled jax-rrtmgp uses a single scalar
   `sfc_alb` for all columns. The AMIP driver passes the column-mean of the
   ice/ocean blended albedo. At coarse resolution (C16-C48) this is a good
   approximation; at fine resolution with large ice fraction gradients it may
   introduce small errors.

2. **Prescribed ozone**: The O3 profile is analytical (Gaussian in
   log-pressure with latitude dependence). It does not vary with season or
   chemistry. A proper ozone climatology (e.g., from CMIP6 forcing files)
   would improve stratospheric heating further.

3. **No aerosols**: Aerosol direct and indirect effects are absent.

4. **Uniform well-mixed gases**: CO2, CH4, N2O are spatially and temporally
   uniform by default. Time-varying concentrations from CMIP forcing files are
   supported via `GHGConfig(source="file", path="...")` with NetCDF time
   interpolation (`ExternalForcingConfig`), but the AMIP driver currently uses
   constant values from CLI flags.

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

### Production 10-year AMIP

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
| RRTMG radiation | **Active** | Correlated-k, H2O+CO2+CH4+N2O+O3, clear-sky |
| SBM convection | **Active** | Frierson 2007 |
| BL exchange | **Active** | Bulk aerodynamic (constant default); MOST/COARE3/LY04 available |
| Large-scale condensation | **Active** | Saturation adjustment |
| Rayleigh friction | **Active** | BL + free-atmosphere drag |
| Checkpoint/restart | **Active** | NPZ-based, reproducible |
| Experiment config | **Active** | JSON-serializable `AMIPExperimentConfig` |
| Diurnal cycle | **Active** | Instantaneous cos(SZA) per column; `--diurnal-cycle` |
| Cloud-radiation coupling | **Active** | Sundqvist or Xu-Randall cloud fraction → RRTMG optics |
| Microphysics | **Active** | Kessler warm-rain or Sundqvist; `--microphysics` |
| Ozone | **Active** | Standard (US Std Atm) or analytical (lat-dependent); `--ozone-source` |
| Dynamic albedo | **Active** | Temperature/zenith-dependent ice+snow albedo; `--dynamic-albedo` |
| Energy budget | **Active** | Online column energy, TOA balance, residual tracking |
| Monthly means | **Active** | Zonal-mean and global-mean monthly accumulation; `--monthly-means` |
| GHG time-varying | **Active** | `GHGConfig(source="file")` with NetCDF time interpolation |
| Experiment templates | **Active** | piControl, historical, SSP2-4.5, SSP5-8.5, AMIP, 1pctCO₂; `create_experiment_config()` |
| CMOR output | **Active** | CF-1.8/CMIP6 DRS NetCDF via `CFWriter`; 27 CMOR variables |
| Restart/reproducibility | **Active** | SHA-256 state digests, config hashes; `verify_reproducibility()` |
| Tuning validation | **Active** | `validate_tuning()` checks CFL, ranges, conflicts |
| Ozone from file | **Active** | `OzoneConfig(enabled=True)` monthly zonal-mean from NetCDF; not yet connected to radiation |
| Aerosol from file | **Active** | `AerosolConfig(enabled=True)` monthly zonal-mean from NetCDF; not yet connected to radiation |
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
--resolution INT                  Cubed-sphere N (default: 16)
--nlev INT                        Number of vertical levels (default: 40)
--dt FLOAT                        Time step [seconds] (default: 600)
--days INT                        Integration length [days] (default: 200)
--start-day FLOAT                 Start day within forcing record
--diag-days INT                   Diagnostic output interval [days] (default: 5)

# Forcing
--dataset {cobe,hadisst,custom}   Forcing dataset preset
--forcing-path PATH               Path to NetCDF forcing file (required unless restarting)
--sst-var NAME                    SST variable name (custom only)
--sic-var NAME                    SIC variable name (custom only)
--sst-offset FLOAT                Additive offset for SST (e.g., 273.15)
--sic-scale FLOAT                 Multiplicative scale for SIC (e.g., 0.01)

# Radiation
--radiation {gray,rrtmg}          Radiation scheme (default: gray)
--rad-update-steps INT            Radiation call frequency [steps] (default: 1)
--diurnal-cycle                   Use instantaneous solar zenith angle (default: off)
--co2-ppmv FLOAT                  CO2 concentration [ppmv] for RRTMG (default: 415)
--ch4-ppbv FLOAT                  CH4 concentration [ppbv] for RRTMG (default: 1900)
--n2o-ppbv FLOAT                  N2O concentration [ppbv] for RRTMG (default: 332)
--ozone-source {standard,analytical,none}  Ozone profile (default: standard)
--clouds {none,sundqvist,xu_randall}       Cloud fraction scheme (default: none)

# Physics
--microphysics {none,kessler,sundqvist}    Microphysics scheme (default: none)
--dynamic-albedo                  Temperature/zenith-dependent surface albedo (default: off)

# Diagnostics
--monthly-means                   Accumulate zonal/global monthly means (default: off)

# I/O
--output PATH                     Output directory
--checkpoint-days INT             Checkpoint interval [days]; 0 = off (default)
--restart-from PATH               Restart from checkpoint .npz file
```

## Post-run validation

The `scripts/validate_amip.py` script checks run output against observational
targets:

```bash
python scripts/validate_amip.py results/amip_run/ --spinup 365
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
