# CMIP Readiness Status

Status of legoESM components for CMIP-class production experiments.

## Surface Coupler

| Feature | Status | Notes |
|---|---|---|
| Tile-based coupling (ocean/ice/land/lake) | Done | Area-weighted blending, flux accumulator |
| Ocean albedo (constant) | Done | `CouplerConfig.ocean_albedo` honoured |
| Ocean albedo (zenith-dependent) | Done | Briegleb 1992 via `OceanAlbedoConfig(method="zenith")` |
| Snow albedo feedback | Done | Age-dependent, latitude-varying |
| Ice albedo (temperature-dependent) | Done | Via `SeaIceConfig.temp_dependent_albedo` |
| Bulk flux schemes | Done | Constant, COARE 3.0, Large & Yeager 2004 |
| q_surface consistency | Done | Recomputed from updated T in all slab tiles |
| Coupling accumulator / asynchronous coupling | Done | Window-mean with residual carry |

## External Forcing

| Feature | Status | Notes |
|---|---|---|
| GHG concentrations (constant) | Done | CO2, CH4, N2O via `GHGConfig` |
| GHG concentrations (time-varying from file) | Done | NetCDF interpolation in `get_ghg_at_time()` |
| Ozone climatology (monthly zonal-mean) | Done | NetCDF loading + cyclic interpolation |
| Aerosol optical depth climatology | Done | NetCDF loading + cyclic interpolation |
| Total solar irradiance (constant) | Done | Via `SolarConfig.S_0` |
| Total solar irradiance (time-varying from file) | Done | NetCDF interpolation in `get_tsi_at_time()` |
| Ozone connected to radiation | Done | Analytical + standard profiles, passed to RRTMGP via `o3_vmr` |
| Aerosol connected to radiation | Done | AOD climatology loaded and applied |
| Volcanic forcing | Done | Stratospheric AOD perturbation support |
| Land-use change forcing | Missing | Static land fraction only |

## Radiation

| Feature | Status | Notes |
|---|---|---|
| Gray radiation (Held-Suarez-like) | Done | |
| RRTMGP (multi-band) | Done | Gas optics, cloud optics |
| Diurnal cycle | Done | Optional via `RadiationConfig.diurnal_cycle` |
| Spectral solar distribution | Missing | Single TSI value only, no spectral bands |
| Interactive aerosol-radiation coupling | Missing | |

## Atmosphere Dynamics

| Feature | Status | Notes |
|---|---|---|
| Cubed-sphere (hydrostatic, nonhydrostatic) | Done | C16-C64 tested |
| Spectral transform (T21-T85) | Done | |
| Centered, FV, FC-Gram, FC-Gram C-grid discretizations | Done | |
| Hyperdiffusion | Done | |
| Sponge layer | Done | |

## Ocean

| Feature | Status | Notes |
|---|---|---|
| Fixed SST (AMIP-style) | Done | Constant or spatial map |
| Slab ocean (mixed layer) | Done | Freezing clamp |
| Two-layer ocean | Done | Deep restoring option |
| 3D ocean dynamics (multiple discretizations) | Done | Centered, FV, FC-Gram, FC-Gram C-grid |
| Ocean biogeochemistry (abiotic) | Done | DIC + ALK, carbonate equilibria, air-sea CO₂ (Wanninkhof 2014) |
| Ocean biogeochemistry (NPZD) | Done | N-P-Z-D ecosystem + Redfield coupling to DIC/ALK |
| Ocean tracer transport | Done | FV tracer option |

## Sea Ice

| Feature | Status | Notes |
|---|---|---|
| Thermodynamic slab ice | Done | Growth/melt, conductive flux |
| Ice concentration (prognostic) | Done | |
| Ice velocity (free drift) | Done | Diagnostic |
| EVP rheology / dynamics | Done | Hunke & Dukowicz 1997, subcycled momentum, `dynamics="evp"` |
| Multi-category ice | Done | CICE framework, Lipscomb 2001 linear remapping, `n_categories > 1` |

## Land Surface

| Feature | Status | Notes |
|---|---|---|
| Slab land (thermal + bucket hydrology) | Done | |
| Multilayer land | Done | Richards equation (6 retention curves) + Johansen thermal diffusion |
| Snow budget (accumulation/melt) | Done | Age-dependent albedo, latitude-varying vegetation |
| Vegetation / stomatal conductance | Done | Farquhar + Ball-Berry/Medlyn (coupled), Jarvis (uncoupled) |
| Land carbon cycle | Done | DALEC-990 6-pool + seasonal scheme; GPP via Farquhar or LUE |
| Land stability validation | Done | 32 tests: energy/water budget closure, seasonal behavior, 180-day runs |

## Lake

| Feature | Status | Notes |
|---|---|---|
| Two-layer lake (epilimnion + hypolimnion) | Done | Wind-enhanced mixing |

## Diagnostics

| Feature | Status | Notes |
|---|---|---|
| Energy budget tracking | Done | Column MSE, TOA/surface fluxes, dE/dt residual |
| Monthly-mean accumulation | Done | Zonal means, 3D profiles, global scalars |
| AMIP validation targets | Done | T_2m, precip, OLR, TOA imbalance, jet position, ITCZ |

## Parallelism

| Feature | Status | Notes |
|---|---|---|
| Cubed-sphere face sharding | Done | 1-6 devices |
| Sub-face tiling | Done | Multiples of 6 devices |
| Lat-lon / level sharding | Done | Domain decomposition |
| MPI halo exchange (structured) | Done | mpi4jax, validated with `mpirun -np 2/3/6` |
| Voronoi mesh decomposition | Done | RCB + METIS, 2-layer halo, entity ownership |
| Voronoi halo exchange (MPI) | Done | mpi4jax sendrecv, entity-type tagging |
| Ensemble parallelism | Done | vmap + NamedSharding + scan, gradient checkpointing |
| Apple Silicon Metal | Done | Metal/CPU hybrid routing |

## CMIP Infrastructure

| Feature | Status | Notes |
|---|---|---|
| CMOR/CF-compliant output | Done | `CFWriter` with CF-1.8, CMIP6 DRS naming, 27 variables (Amon + Lmon) |
| Experiment templates | Done | piControl, historical, SSP2-4.5, SSP5-8.5, AMIP, 1pctCO₂ |
| Built-in GHG time series | Done | Linear interpolation for historical + SSP scenarios |
| Restart/reproducibility | Done | SHA-256 state digests, config hashes, platform metadata |
| Tuning guide | Done | 16 parameters, validation, resolution-appropriate defaults |

## Remaining Gaps for CMIP Production

1. **Land-use change**: Static land mask, no transient land cover.
2. **Dynamic vegetation**: No interactive LAI (currently prescribed).
3. **Spectral solar distribution**: Only broadband TSI, no spectral bands.
4. **Interactive aerosol-cloud coupling**: AOD affects radiation but not cloud droplet number concentration.
5. **Ice sheet dynamics**: No shallow-ice/shallow-shelf approximation.
6. **Atmospheric chemistry**: No prognostic CH4/N2O/CFC/O3 tracers (prescribed only).
