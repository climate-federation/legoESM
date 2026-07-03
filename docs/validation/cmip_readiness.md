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
| 3D ocean dynamics (multiple discretizations) | Done | Centered, FV, FC-Gram, FC-Gram C-grid, MPAS Voronoi |
| Tripolar grid (eORCA1) | Done | NEMO mesh_mask loader + tensor pole-fold halo (`grids/tripole.py`) |
| Cross-grid metric consistency | Done | Ocean test matrix 57/57 PASS across lat-lon / tripolar / cubed-sphere / MPAS |
| Realistic bathymetry | Done | ETOPO/GEBCO/ERDDAP + Laplacian smoothing + MEO r-cap + flood-fill basin removal |
| Ice-shelf cavity coupling | Done | Holland & Jenkins (1999) basal melt at ice base; lat-lon + MPAS apply paths |
| Tidal mixing | Done | Jayne & St-Laurent (2001) abyssal K with tracer-mixing integration |
| River runoff | Done | Dai–Trenberth point→grid projection |
| OMIP-2 SSS restoring | Done | WOA SSS climatology loader + FreshwaterForcing channel |
| OMIP-2 atmospheric forcing | Done | JRA55-do RYF, float32 preload for 32 GB GPUs |
| External τ / q_net / sw_down pathway | Done | C-grid PE supports prescribed surface forcing |
| AMOC@26.5°N diagnostic | Done | `ocean.spinup.compute_amoc_timeseries` |
| Centennial spin-up workflow | Done | Auto-restart, RPE / volume drift, Bryan–Lewis acceleration |
| Mass conservation projection | Done | Lat-lon and MPAS implicit barotropic solvers |
| Ocean biogeochemistry (abiotic) | Done | DIC + ALK, carbonate equilibria, air-sea CO₂ (Wanninkhof 2014) |
| Ocean biogeochemistry (NPZD) | Done | N-P-Z-D ecosystem + Redfield coupling to DIC/ALK |
| Ocean tracer transport | Done | FV / PPM-FCT (Zalesak) / TVD; DST-3 vertical advection |
| Ocean lateral mixing | Done | Harmonic, biharmonic, GM-Redi, Visbeck adaptive-GM, Leith viscosity, backscatter |
| Ocean fidelity vs Veros | Done | `ocean/fidelity/` Veros DINO / Eady adapters and comparison reports |

## Sea Ice

| Feature | Status | Notes |
|---|---|---|
| Thermodynamic slab ice | Done | Growth/melt, conductive flux |
| Ice concentration (prognostic) | Done | |
| Ice velocity (free drift) | Done | Diagnostic |
| EVP rheology / dynamics | Done | Hunke & Dukowicz 1997, subcycled momentum, `dynamics="evp"` |
| mEVP rheology | Done | Bouillon 2013 / Kimmritz 2015 pseudo-time relaxation, `dynamics="mevp"` |
| MPAS Voronoi sea-ice support | Done | Transport + rheology + dynamics + no-flux boundary on icosahedral mesh |
| Tensor pole-fold halo (tripolar) | Done | Lat-lon dispatch documented in `ice/rheology.py` |
| Multi-category ice | Done | CICE framework, Lipscomb 2001 linear remapping, `n_categories > 1` |
| Snow on ice | Done | `ice/snow.py` |
| Brine pockets | Done | `ice/brine.py` |
| Ridging | Done | `ice/ridging.py` |
| Delta-Eddington shortwave | Done | `ice/shortwave.py` |
| Melt ponds | Done | `ice/ponds.py` |

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
| Voronoi mesh decomposition | Done | `auto` (METIS when `pymetis` present, else RCB) + Hilbert SFC, 2-layer halo, owned-first entity ownership |
| Voronoi halo exchange (MPI) | Done | mpi4jax sendrecv, entity-type tagging |
| Ensemble parallelism | Done | vmap + NamedSharding + scan, gradient checkpointing |
| Apple Silicon Metal | Done | Metal/CPU hybrid routing |

## CMIP Infrastructure

| Feature | Status | Notes |
|---|---|---|
| CMOR/CF-compliant output | Done | `CFWriter` + `io/cmor_output.py` with CF-1.8, CMIP6 DRS naming |
| CMIP6 tables wired | Done | **Amon, Lmon, Omon, Oyr, Ofx, SImon, SIyr** |
| Ocean overturning + OSNAP transports | Done | Per the CMIP6 diagnostic expansion |
| Global ocean / sea-ice scalars | Done | Variance, dianeutral mixing, mass, heat, salt |
| Experiment templates | Done | piControl, historical, SSP2-4.5, SSP5-8.5, AMIP, 1pctCO₂ |
| Built-in GHG time series | Done | Linear interpolation for historical + SSP scenarios |
| Restart/reproducibility | Done | SHA-256 state digests, config hashes, platform metadata |
| Persistent JAX JIT cache | Done | Issue #273; opt-out via `LEGOESM_JAX_CACHE_DISABLE=1` |
| SPMD halo backend (AMIP production) | Done | Issue #275 |
| Tuning guide | Done | 16 parameters, validation, resolution-appropriate defaults |

## Remaining Gaps for CMIP Production

1. **Land-use change**: Static land mask, no transient land cover.
2. **Dynamic vegetation**: No interactive LAI (currently prescribed).
3. **Spectral solar distribution**: Only broadband TSI, no spectral bands.
4. **Interactive aerosol-cloud coupling**: AOD affects radiation but not cloud droplet number concentration.
5. **Ice sheet dynamics**: No shallow-ice/shallow-shelf approximation.
6. **Atmospheric chemistry**: No prognostic CH4/N2O/CFC/O3 tracers (prescribed only).
