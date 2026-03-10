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
| Ozone connected to radiation | Missing | Ozone data loaded but not yet passed to radiation schemes |
| Aerosol connected to radiation | Missing | AOD data loaded but not yet modifying SW fluxes |
| Volcanic forcing | Missing | No eruption-driven AOD or stratospheric heating |
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
| Ocean biogeochemistry | Missing | No carbon cycle / DIC |
| Ocean tracer transport | Done | FV tracer option |

## Sea Ice

| Feature | Status | Notes |
|---|---|---|
| Thermodynamic slab ice | Done | Growth/melt, conductive flux |
| Ice concentration (prognostic) | Done | |
| Ice velocity (free drift) | Done | Diagnostic |
| Rheology / dynamics | Missing | No VP or EVP solver |
| Multi-category ice | Missing | Single thickness category |

## Land Surface

| Feature | Status | Notes |
|---|---|---|
| Slab land (thermal + bucket hydrology) | Done | |
| Multilayer land | Done | Multi-layer soil temperature |
| Snow budget (accumulation/melt) | Done | |
| Vegetation / canopy model | Missing | No stomatal conductance or LAI |
| Land carbon cycle | Missing | |

## Lake

| Feature | Status | Notes |
|---|---|---|
| Two-layer lake (epilimnion + hypolimnion) | Done | Wind-enhanced mixing |

## Remaining Gaps for CMIP Production

1. **Aerosol-radiation coupling**: AOD is loaded but not applied to SW radiation.
2. **Ozone-radiation coupling**: Ozone is loaded but not passed to radiation.
3. **Volcanic forcing**: No stratospheric aerosol injection events.
4. **Land-use change**: Static land mask, no transient land cover.
5. **Ocean biogeochemistry**: No carbon cycle or ocean CO2 flux feedback.
6. **Dynamic vegetation**: No interactive LAI or stomatal conductance.
7. **Sea-ice dynamics**: Thermodynamics only, no rheological solver.
8. **Multi-category sea ice**: Single thickness category.
9. **Spectral solar distribution**: Only broadband TSI, no spectral bands.
