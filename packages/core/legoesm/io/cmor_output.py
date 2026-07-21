"""CF/CMOR-compliant NetCDF output pipeline for legoESM.

Writes model output as CF-1.8 / CMOR 3.x compliant NetCDF4 files,
suitable for submission to CMIP6-class model intercomparisons.  Each
variable is stored in its own file following the CMIP6 DRS:

    <var>_<table>_<model>_<experiment>_<variant>_<grid>_<time-range>.nc

The module is self-contained: it handles CMOR variable metadata
(standard_name, long_name, units, cell_methods), CF coordinate
attributes, time axis with bounds, and global attributes.

Dependencies
------------
- xarray (lazy import)
- netCDF4 (lazy import, used only for compression tuning if needed)
- numpy (for JAX array conversion)

Usage
-----
    from legoesm.io.cmor_output import CFWriter

    writer = CFWriter(
        output_dir="output/cmor",
        experiment_id="amip",
        model_id="legoESM-1-0",
        freq="mon",
        calendar="noleap",
        ref_date="0001-01-01",
    )

    writer.write_field(
        var_name="tas",
        data=t2m_array,          # numpy or JAX array, shape (nlat, nlon)
        time=15.0,               # days since ref_date
        time_bounds=(0.0, 30.0),
        lat=lat_1d,
        lon=lon_1d,
    )

    writer.write_monthly(monthly_data, lat_1d, lon_1d, plev=plev_pa)
    writer.close()

Notes
-----
- JAX arrays are converted to numpy before writing.
- Cubed-sphere to lat-lon regridding is NOT performed here; the caller
  should regrid via ``legoesm.grids.regridding`` before passing data.

References
----------
- CF Conventions 1.8: https://cfconventions.org/
- CMOR 3 specification: https://cmor.llnl.gov/
- CMIP6 data reference syntax (DRS):
  https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6aA
"""

from __future__ import annotations

import datetime
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import numpy as np


# Standard CMIP6 license text (Creative Commons Attribution 4.0 International)
CMIP6_LICENSE = (
    "CMIP6 model data produced by legoESM is licensed under a "
    "Creative Commons Attribution 4.0 International License "
    "(https://creativecommons.org/licenses/). Consult "
    "https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use "
    "governing CMIP6 output, including citation requirements and "
    "proper acknowledgment."
)

# Realm assignment for each CMOR table (CMIP6 CV: required_global_attributes)
_TABLE_REALM: Dict[str, str] = {
    "Amon": "atmos",
    "day": "atmos",
    "Aday": "atmos",   # legacy alias — "day" is the CMIP6 CV value
    "Lmon": "land",
    "Omon": "ocean",
    "Oyr": "ocean",
    "Ofx": "ocean",
    "SImon": "seaIce",
    "SIyr": "seaIce",
    "fx": "atmos",
}

# CMIP6 surface-field reference heights [m].  These variables are defined
# at a fixed height above the surface rather than at a model level, so they
# carry a scalar ``height`` coordinate per CMIP6 spec.
_VAR_REFERENCE_HEIGHT: Dict[str, float] = {
    # 2 m air-temperature / humidity diagnostics
    "tas": 2.0,
    "tasmin": 2.0,
    "tasmax": 2.0,
    "huss": 2.0,
    "hurs": 2.0,
    # 10 m wind diagnostics
    "uas": 10.0,
    "vas": 10.0,
    "sfcWind": 10.0,
    "sfcWindmax": 10.0,
}


# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when actually writing files
# ---------------------------------------------------------------------------

def _import_xarray():
    """Lazily import xarray."""
    import xarray as xr
    return xr


def _import_netcdf4():
    """Lazily import netCDF4 (optional, for low-level compression)."""
    try:
        import netCDF4
        return netCDF4
    except ImportError:
        return None


def _to_numpy(arr) -> np.ndarray:
    """Convert a JAX array (or anything array-like) to a numpy ndarray."""
    return np.asarray(arr)


def _resolve_output_dtype(explicit_dtype: str | None) -> np.dtype:
    """Resolve the output dtype for CMOR variable data.

    Priority:
    1. Explicit dtype string (e.g. ``"float64"``) — always wins.
    2. Precision policy's storage dtype — used as a fallback.
       Float32 storage maps to ``np.float32``; anything wider maps
       to ``np.float64``.
    3. ``np.float32`` — safe default if the precision system is
       unavailable.

    Parameters
    ----------
    explicit_dtype : str or None
        If not None, a NumPy dtype string (e.g. ``"float32"``).

    Returns
    -------
    np.dtype
        Resolved NumPy dtype for on-disk variable data.
    """
    if explicit_dtype is not None:
        return np.dtype(explicit_dtype)

    try:
        from legoesm.core.precision import get_policy
        import jax.numpy as jnp

        policy_dtype = get_policy().storage
        if policy_dtype == jnp.float32:
            return np.dtype(np.float32)
        return np.dtype(np.float64)
    except Exception:
        return np.dtype(np.float32)


# =========================================================================
# CMOR Variable Tables
# =========================================================================

# Each entry: {standard_name, long_name, units, cell_methods, dimensions}
# ``dimensions`` is a tuple of coordinate names that follow
# (time, ..., lat, lon).  "plev" marks 3-D pressure-interpolated fields.

_AMON_VARIABLES: Dict[str, Dict[str, str]] = {
    # --- Near-surface / surface scalars ---
    "tas": {
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "units": "K",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "ps": {
        "standard_name": "surface_air_pressure",
        "long_name": "Surface Air Pressure",
        "units": "Pa",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "psl": {
        "standard_name": "air_pressure_at_mean_sea_level",
        "long_name": "Sea Level Pressure",
        "units": "Pa",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "pr": {
        "standard_name": "precipitation_flux",
        "long_name": "Precipitation",
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "prw": {
        "standard_name": "atmosphere_mass_content_of_water_vapor",
        "long_name": "Water Vapor Path",
        "units": "kg m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "clt": {
        "standard_name": "cloud_area_fraction",
        "long_name": "Total Cloud Cover Percentage",
        "units": "%",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "clwvi": {
        "standard_name": "atmosphere_mass_content_of_cloud_condensed_water",
        "long_name": "Condensed Water Path",
        "units": "kg m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "clivi": {
        "standard_name": "atmosphere_mass_content_of_cloud_ice",
        "long_name": "Ice Water Path",
        "units": "kg m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    # --- TOA radiation ---
    "rsdt": {
        "standard_name": "toa_incoming_shortwave_flux",
        "long_name": "TOA Incident Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rsut": {
        "standard_name": "toa_outgoing_shortwave_flux",
        "long_name": "TOA Outgoing Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rlut": {
        "standard_name": "toa_outgoing_longwave_flux",
        "long_name": "TOA Outgoing Longwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rsutcs": {
        "standard_name": "toa_outgoing_shortwave_flux_assuming_clear_sky",
        "long_name": "TOA Outgoing Clear-Sky Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rlutcs": {
        "standard_name": "toa_outgoing_longwave_flux_assuming_clear_sky",
        "long_name": "TOA Outgoing Clear-Sky Longwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    # --- Surface radiation ---
    "rsds": {
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "long_name": "Surface Downwelling Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rlds": {
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "long_name": "Surface Downwelling Longwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rsus": {
        "standard_name": "surface_upwelling_shortwave_flux_in_air",
        "long_name": "Surface Upwelling Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rlus": {
        "standard_name": "surface_upwelling_longwave_flux_in_air",
        "long_name": "Surface Upwelling Longwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    # --- Surface fluxes ---
    "hfss": {
        "standard_name": "surface_upward_sensible_heat_flux",
        "long_name": "Surface Upward Sensible Heat Flux",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "hfls": {
        "standard_name": "surface_upward_latent_heat_flux",
        "long_name": "Surface Upward Latent Heat Flux",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "tauu": {
        "standard_name": "surface_downward_eastward_stress",
        "long_name": "Surface Downward Eastward Wind Stress",
        "units": "Pa",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "tauv": {
        "standard_name": "surface_downward_northward_stress",
        "long_name": "Surface Downward Northward Wind Stress",
        "units": "Pa",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    # --- 3-D atmosphere fields (pressure levels) ---
    "ta": {
        "standard_name": "air_temperature",
        "long_name": "Air Temperature",
        "units": "K",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "ua": {
        "standard_name": "eastward_wind",
        "long_name": "Eastward Wind",
        "units": "m s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "va": {
        "standard_name": "northward_wind",
        "long_name": "Northward Wind",
        "units": "m s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "hus": {
        "standard_name": "specific_humidity",
        "long_name": "Specific Humidity",
        "units": "1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    # --- Additional Amon variables for CMIP submission ---
    "ts": {
        "standard_name": "surface_temperature",
        "long_name": "Surface Temperature",
        "units": "K",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "zg": {
        "standard_name": "geopotential_height",
        "long_name": "Geopotential Height",
        "units": "m",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "wap": {
        "standard_name": "lagrangian_tendency_of_air_pressure",
        "long_name": "Omega (=dp/dt)",
        "units": "Pa s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "hur": {
        "standard_name": "relative_humidity",
        "long_name": "Relative Humidity",
        "units": "%",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "hurs": {
        "standard_name": "relative_humidity",
        "long_name": "Near-Surface Relative Humidity",
        "units": "%",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "clw": {
        "standard_name": "mass_fraction_of_cloud_liquid_water_in_air",
        "long_name": "Mass Fraction of Cloud Liquid Water",
        "units": "1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "cli": {
        "standard_name": "mass_fraction_of_cloud_ice_in_air",
        "long_name": "Mass Fraction of Cloud Ice",
        "units": "1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "plev", "lat", "lon"),
    },
    "rsdscs": {
        "standard_name": "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
        "long_name": "Surface Downwelling Clear-Sky Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rldscs": {
        "standard_name": "surface_downwelling_longwave_flux_in_air_assuming_clear_sky",
        "long_name": "Surface Downwelling Clear-Sky Longwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "evspsbl": {
        "standard_name": "water_evapotranspiration_flux",
        "long_name": "Evaporation Including Sublimation and Transpiration",
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
}

_LMON_VARIABLES: Dict[str, Dict[str, str]] = {
    "gpp": {
        "standard_name": "gross_primary_productivity_of_biomass_"
                         "expressed_as_carbon",
        "long_name": "Gross Primary Production of Biomass "
                     "Expressed as Carbon",
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean area: mean where land",
        "dimensions": ("time", "lat", "lon"),
    },
    "nee": {
        "standard_name": "surface_net_downward_mass_flux_of_carbon_"
                         "dioxide_expressed_as_carbon_due_to_all_"
                         "land_processes_excluding_anthropogenic_"
                         "land_use_change",
        "long_name": "Net Ecosystem Exchange of CO2",
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean area: mean where land",
        "dimensions": ("time", "lat", "lon"),
    },
    "lai": {
        "standard_name": "leaf_area_index",
        "long_name": "Leaf Area Index",
        "units": "1",
        "cell_methods": "time: mean area: mean where land",
        "dimensions": ("time", "lat", "lon"),
    },
    "mrso": {
        "standard_name": "mass_content_of_water_in_soil",
        "long_name": "Total Soil Moisture Content",
        "units": "kg m-2",
        "cell_methods": "time: mean area: mean where land",
        "dimensions": ("time", "lat", "lon"),
    },
    "mrsos": {
        "standard_name": "mass_content_of_water_in_soil_layer",
        "long_name": "Moisture in Upper Portion of Soil Column",
        "units": "kg m-2",
        "cell_methods": "time: mean area: mean where land",
        "dimensions": ("time", "lat", "lon"),
    },
    "tsl": {
        "standard_name": "soil_temperature",
        "long_name": "Temperature of Soil",
        "units": "K",
        "cell_methods": "time: mean area: mean where land",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
}

_OMON_VARIABLES: Dict[str, Dict[str, str]] = {
    # --- 2-D surface ocean (OMIP-2 core, CMIP6 Omon table) ---
    "tos": {
        "standard_name": "sea_surface_temperature",
        "long_name": "Sea Surface Temperature",
        "units": "K",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "sos": {
        "standard_name": "sea_surface_salinity",
        "long_name": "Sea Surface Salinity",
        "units": "0.001",  # CMIP6 dimensionless salinity
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "zos": {
        "standard_name": "sea_surface_height_above_geoid",
        "long_name": "Sea Surface Height Above Geoid",
        "units": "m",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "mlotst": {
        "standard_name": "ocean_mixed_layer_thickness_defined_by_sigma_t",
        "long_name": "Ocean Mixed Layer Thickness Defined by Sigma T",
        "units": "m",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "hfds": {
        "standard_name": "surface_downward_heat_flux_in_sea_water",
        "long_name": "Downward Heat Flux at Sea Water Surface",
        "units": "W m-2",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "wfo": {
        "standard_name": "water_flux_into_sea_water",
        "long_name": "Water Flux Into Sea Water",
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "tauuo": {
        "standard_name": "surface_downward_x_stress",
        "long_name": "Surface Downward X Stress",
        "units": "N m-2",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "tauvo": {
        "standard_name": "surface_downward_y_stress",
        "long_name": "Surface Downward Y Stress",
        "units": "N m-2",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    # --- 3-D ocean (OMIP-2 core).  ``depth`` carries ocean depth
    # (positive down).  The writer attaches OMIP-2 attrs when
    # ``table_id == "Omon"``.
    "thetao": {
        # CMIP6 Omon spec: thetao is reported in degC (NOT Kelvin).
        # Callers writing this variable must convert model temperature
        # (typically held in Kelvin internally) to Celsius before
        # passing it to CFWriter.write_field.
        "standard_name": "sea_water_potential_temperature",
        "long_name": "Sea Water Potential Temperature",
        "units": "degC",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
    "so": {
        "standard_name": "sea_water_salinity",
        "long_name": "Sea Water Salinity",
        "units": "0.001",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
    "uo": {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Sea Water X Velocity",
        "units": "m s-1",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
    "vo": {
        "standard_name": "sea_water_y_velocity",
        "long_name": "Sea Water Y Velocity",
        "units": "m s-1",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
    # --- Meridional overturning streamfunction (CMIP6 Omon).
    # ``basin`` is a CMIP6 named-coordinate dimension with three values
    # (global, atlantic_arctic, indian_pacific); writer expands it as
    # an integer index axis with attached string coordinate variable.
    "msftmz": {
        "standard_name": "ocean_meridional_overturning_mass_streamfunction",
        "long_name": "Ocean Meridional Overturning Mass Streamfunction",
        "units": "kg s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "basin", "depth", "lat"),
    },
    "msftyz": {
        # Mass-stream-function on geometric depth coordinate.
        "standard_name": "ocean_y_overturning_mass_streamfunction",
        "long_name": "Ocean Y Overturning Mass Streamfunction",
        "units": "kg s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "basin", "depth", "lat"),
    },
    # --- Global mean scalars (single-point per timestep).  ``dimensions
    # = ("time",)`` triggers the CMOR ``zg=0`` integer-axis placeholder
    # in CFWriter so the field carries the correct CMIP6 metadata.
    "tosga": {
        "standard_name": "sea_surface_temperature",
        "long_name": "Global Average Sea Surface Temperature",
        "units": "K",
        "cell_methods": "area: mean where sea time: mean",
        "dimensions": ("time",),
    },
    "sosga": {
        "standard_name": "sea_surface_salinity",
        "long_name": "Global Average Sea Surface Salinity",
        "units": "0.001",
        "cell_methods": "area: mean where sea time: mean",
        "dimensions": ("time",),
    },
    "thetaoga": {
        "standard_name": "sea_water_potential_temperature",
        "long_name": "Global Average Sea Water Potential Temperature",
        "units": "degC",
        "cell_methods": "area: mean volume: mean where sea time: mean",
        "dimensions": ("time",),
    },
    "soga": {
        "standard_name": "sea_water_salinity",
        "long_name": "Global Mean Sea Water Salinity",
        "units": "0.001",
        "cell_methods": "area: mean volume: mean where sea time: mean",
        "dimensions": ("time",),
    },
    "masso": {
        "standard_name": "sea_water_mass",
        "long_name": "Sea Water Mass",
        "units": "kg",
        "cell_methods": "area: sum where sea time: mean",
        "dimensions": ("time",),
    },
    "volo": {
        "standard_name": "sea_water_volume",
        "long_name": "Sea Water Volume",
        "units": "m3",
        "cell_methods": "area: sum where sea time: mean",
        "dimensions": ("time",),
    },
    "wo": {
        # Sign convention: ``standard_name = upward_sea_water_velocity``
        # — values are POSITIVE UPWARD.  Ocean models that carry
        # vertical velocity positive-downward on a depth-positive-down
        # vertical coordinate (e.g. legoESM's z* core) MUST negate the
        # field before passing it to CFWriter.write_field.
        "standard_name": "upward_sea_water_velocity",
        "long_name": "Upward Sea Water Velocity",
        "units": "m s-1",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
    "rhopoto": {
        "standard_name": "sea_water_potential_density",
        "long_name": "Sea Water Potential Density",
        "units": "kg m-3",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "depth", "lat", "lon"),
    },
    # --- Sea ice (kept under Omon for backward-compat; CMIP6 places
    # ``siconc``/``sithick``/``siu``/``siv`` in SImon — see
    # ``_SIMON_VARIABLES`` below).
    "sic": {
        "standard_name": "sea_ice_area_fraction",
        "long_name": "Sea-Ice Area Percentage (Ocean Grid)",
        "units": "%",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
}


# --- CMIP6 SImon (sea-ice monthly) — the OMIP-2 protocol points here
# for the sea-ice prognostic variables produced by the ocean / coupled
# experiments.  All fields live on the ocean grid.
_SIMON_VARIABLES: Dict[str, Dict[str, str]] = {
    "siconc": {
        "standard_name": "sea_ice_area_fraction",
        "long_name": "Sea-Ice Area Fraction",
        "units": "%",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "sithick": {
        # CMIP6 SImon cell_methods string is the canonical CF form
        # ``time: mean area: mean where sea_ice``; do not append
        # mask references in parentheses (CMOR/CF validators reject
        # those).  The mask is conveyed by the CMIP6 ``ancillary``
        # mechanism, not by cell_methods.
        "standard_name": "sea_ice_thickness",
        "long_name": "Sea Ice Thickness",
        "units": "m",
        "cell_methods": "time: mean area: mean where sea_ice",
        "dimensions": ("time", "lat", "lon"),
    },
    # Sea-ice velocity components.  CMIP6 SImon expects
    # eastward/northward components on the **lat-lon** output grid —
    # native curvilinear (cubed-sphere, MPAS, tripole) vectors MUST be
    # rotated to the geographic east/north basis before writing.
    "siu": {
        "standard_name": "sea_ice_x_velocity",
        "long_name": "X-Component of Sea-Ice Velocity",
        "units": "m s-1",
        "cell_methods": "time: mean area: mean where sea_ice",
        "dimensions": ("time", "lat", "lon"),
    },
    "siv": {
        "standard_name": "sea_ice_y_velocity",
        "long_name": "Y-Component of Sea-Ice Velocity",
        "units": "m s-1",
        "cell_methods": "time: mean area: mean where sea_ice",
        "dimensions": ("time", "lat", "lon"),
    },
    "sitemptop": {
        "standard_name": "sea_ice_surface_temperature",
        "long_name": "Surface Temperature of Sea Ice",
        "units": "K",
        "cell_methods": "time: mean area: mean where sea_ice",
        "dimensions": ("time", "lat", "lon"),
    },
}

_DAY_VARIABLES: Dict[str, Dict[str, str]] = {
    "tas": {
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "units": "K",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "tasmin": {
        "standard_name": "air_temperature",
        "long_name": "Daily Minimum Near-Surface Air Temperature",
        "units": "K",
        "cell_methods": "time: minimum",
        "dimensions": ("time", "lat", "lon"),
    },
    "tasmax": {
        "standard_name": "air_temperature",
        "long_name": "Daily Maximum Near-Surface Air Temperature",
        "units": "K",
        "cell_methods": "time: maximum",
        "dimensions": ("time", "lat", "lon"),
    },
    "pr": {
        "standard_name": "precipitation_flux",
        "long_name": "Precipitation",
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "psl": {
        "standard_name": "air_pressure_at_mean_sea_level",
        "long_name": "Sea Level Pressure",
        "units": "Pa",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rsut": {
        "standard_name": "toa_outgoing_shortwave_flux",
        "long_name": "TOA Outgoing Shortwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "rlut": {
        "standard_name": "toa_outgoing_longwave_flux",
        "long_name": "TOA Outgoing Longwave Radiation",
        "units": "W m-2",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "ua850": {
        "standard_name": "eastward_wind",
        "long_name": "Eastward Wind at 850 hPa",
        "units": "m s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
    "va850": {
        "standard_name": "northward_wind",
        "long_name": "Northward Wind at 850 hPa",
        "units": "m s-1",
        "cell_methods": "time: mean",
        "dimensions": ("time", "lat", "lon"),
    },
}

# ``fx`` table — time-invariant fields. CF ``cell_methods`` and ``time``
# dimension are intentionally absent; these files are written once per run.
_FX_VARIABLES: Dict[str, Dict[str, str]] = {
    "orog": {
        "standard_name": "surface_altitude",
        "long_name": "Surface Altitude",
        "units": "m",
        "cell_methods": "area: mean",
        "dimensions": ("lat", "lon"),
    },
    "areacella": {
        "standard_name": "cell_area",
        "long_name": "Grid-Cell Area for Atmospheric Grid Variables",
        "units": "m2",
        "cell_methods": "area: sum",
        "dimensions": ("lat", "lon"),
    },
    "sftlf": {
        "standard_name": "land_area_fraction",
        "long_name": "Land Area Fraction",
        "units": "%",
        "cell_methods": "area: mean",
        "dimensions": ("lat", "lon"),
    },
}

# --- CMIP6 ``Oyr`` (annual ocean) — same variables as ``Omon`` but
# accumulated to annual means.  Production OMIP-2 + CMIP6 archive
# expects long centennial integrations to ship ``Oyr`` rather than
# ``Omon`` for the deep / global-mean diagnostics.  Reuses the
# ``Omon`` variable specs verbatim — only the table_id differs (which
# the writer uses to build the filename + DRS path).
_OYR_VARIABLES: Dict[str, Dict[str, str]] = dict(_OMON_VARIABLES)


# --- CMIP6 ``SIyr`` (annual sea ice) — annual means of ``SImon``.
_SIYR_VARIABLES: Dict[str, Dict[str, str]] = dict(_SIMON_VARIABLES)


# --- CMIP6 ``Ofx`` (time-invariant ocean fields).  Bathymetry, cell
# area, sea-area fraction, and per-layer mass / volume of the
# reference state.  Written once per run.
_OFX_VARIABLES: Dict[str, Dict[str, str]] = {
    "areacello": {
        "standard_name": "cell_area",
        "long_name": "Grid-Cell Area for Ocean Variables",
        "units": "m2",
        "cell_methods": "area: sum",
        "dimensions": ("lat", "lon"),
    },
    "deptho": {
        "standard_name": "sea_floor_depth_below_geoid",
        "long_name": "Sea Floor Depth Below Geoid",
        "units": "m",
        "cell_methods": "area: mean where sea",
        "dimensions": ("lat", "lon"),
    },
    "sftof": {
        "standard_name": "sea_area_fraction",
        "long_name": "Sea Area Fraction",
        "units": "%",
        "cell_methods": "area: mean",
        "dimensions": ("lat", "lon"),
    },
    "masscello": {
        "standard_name": "sea_water_mass_per_unit_area",
        "long_name": "Sea Water Mass Per Unit Area",
        "units": "kg m-2",
        "cell_methods": "area: mean where sea time: mean",
        "dimensions": ("depth", "lat", "lon"),
    },
    "volcello": {
        "standard_name": "ocean_volume",
        "long_name": "Ocean Grid-Cell Volume",
        "units": "m3",
        "cell_methods": "area: sum where sea time: mean",
        "dimensions": ("depth", "lat", "lon"),
    },
    "thkcello": {
        "standard_name": "cell_thickness",
        "long_name": "Ocean Model Cell Thickness",
        "units": "m",
        "cell_methods": "area: mean where sea time: mean",
        "dimensions": ("depth", "lat", "lon"),
    },
}


# Combined lookup for convenience. ``Aday`` is a legacy alias for
# ``day`` (the CMIP6 CV value); keep both so pre-existing tests and
# callers continue to work.
CMOR_TABLES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Amon": _AMON_VARIABLES,
    "Lmon": _LMON_VARIABLES,
    "Omon": _OMON_VARIABLES,
    "Oyr": _OYR_VARIABLES,
    "Ofx": _OFX_VARIABLES,
    "SImon": _SIMON_VARIABLES,
    "SIyr": _SIYR_VARIABLES,
    "day": _DAY_VARIABLES,
    "Aday": _DAY_VARIABLES,
    "fx": _FX_VARIABLES,
}

# Standard CMIP6 pressure levels [Pa] (19 levels, top-to-bottom)
CMIP6_PLEV19 = np.array([
    100_000.0, 92_500.0, 85_000.0, 70_000.0, 60_000.0,
    50_000.0, 40_000.0, 30_000.0, 25_000.0, 20_000.0,
    15_000.0, 10_000.0, 7_000.0, 5_000.0, 3_000.0,
    2_000.0, 1_000.0, 500.0, 100.0,
])


# =========================================================================
# Helper: look up CMOR metadata for a variable name
# =========================================================================

def lookup_cmor_entry(
    var_name: str,
    table: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """Return ``(table_id, entry_dict)`` for a CMOR variable name.

    Parameters
    ----------
    var_name : str
        Short CMOR variable name (e.g. ``"tas"``).
    table : str, optional
        Force a specific table (``"Amon"`` or ``"Lmon"``).  If *None*,
        searches all tables and returns the first match.

    Raises
    ------
    KeyError
        If *var_name* is not found in any table.
    """
    if table is not None:
        tbl = CMOR_TABLES.get(table)
        if tbl is None:
            raise KeyError(f"Unknown CMOR table: {table!r}")
        if var_name not in tbl:
            raise KeyError(
                f"Variable {var_name!r} not in table {table!r}"
            )
        return table, tbl[var_name]

    for tbl_id, tbl in CMOR_TABLES.items():
        if var_name in tbl:
            return tbl_id, tbl[var_name]
    raise KeyError(
        f"Variable {var_name!r} not found in any CMOR table"
    )


# =========================================================================
# Coordinate builders
# =========================================================================

def _make_lat_da(lat: np.ndarray):
    """Build a CF-compliant latitude DataArray."""
    xr = _import_xarray()
    return xr.DataArray(
        np.asarray(lat, dtype=np.float64),
        dims=("lat",),
        attrs={
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
            "bounds": "lat_bnds",
        },
    )


def _make_lon_da(lon: np.ndarray):
    """Build a CF-compliant longitude DataArray."""
    xr = _import_xarray()
    return xr.DataArray(
        np.asarray(lon, dtype=np.float64),
        dims=("lon",),
        attrs={
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
            "bounds": "lon_bnds",
        },
    )


def _cell_bounds_from_centers(
    centers: np.ndarray,
) -> np.ndarray:
    """Compute cell edges from cell centers.

    For a uniform-spacing grid the edges sit halfway between neighboring
    centers; the outermost edges are extrapolated by the same half-step.
    Returns a ``(n, 2)`` array of ``(lower_edge, upper_edge)`` pairs.

    Note: this does not wrap modulo 360 for circular axes (e.g. longitude);
    callers are responsible for providing centers on a canonical interval.
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        # Degenerate: use a nominal 1-unit wide cell centered on the point.
        half = 0.5
        return np.array([[c[0] - half, c[0] + half]], dtype=np.float64)
    mids = 0.5 * (c[:-1] + c[1:])
    lower_first = c[0] - (mids[0] - c[0])
    upper_last = c[-1] + (c[-1] - mids[-1])
    edges = np.concatenate([[lower_first], mids, [upper_last]])
    return np.stack([edges[:-1], edges[1:]], axis=-1)


def _make_lat_bnds_da(lat: np.ndarray):
    """Build a latitude cell-bounds DataArray, shape ``(nlat, 2)``.

    Edges are clipped to ``[-90, 90]`` so that polar cells do not extend
    off the sphere, which would otherwise fail CF/CMIP validation.
    """
    xr = _import_xarray()
    bnds = _cell_bounds_from_centers(np.asarray(lat, dtype=np.float64))
    bnds = np.clip(bnds, -90.0, 90.0)
    return xr.DataArray(
        bnds,
        dims=("lat", "bnds"),
        attrs={"units": "degrees_north"},
    )


def _make_lon_bnds_da(lon: np.ndarray):
    """Build a longitude cell-bounds DataArray, shape ``(nlon, 2)``."""
    xr = _import_xarray()
    bnds = _cell_bounds_from_centers(np.asarray(lon, dtype=np.float64))
    return xr.DataArray(
        bnds,
        dims=("lon", "bnds"),
        attrs={"units": "degrees_east"},
    )


def _make_height_da(height_m: float):
    """Build a scalar reference-height coordinate (CMIP6 tas/uas/…)."""
    xr = _import_xarray()
    return xr.DataArray(
        np.float64(height_m),
        attrs={
            "standard_name": "height",
            "long_name": "height",
            "units": "m",
            "axis": "Z",
            "positive": "up",
        },
    )


def _make_plev_da(plev: np.ndarray):
    """Build a CF-compliant pressure-level DataArray.

    Parameters
    ----------
    plev : array-like
        Pressure levels in Pa, ordered top-to-bottom (ascending pressure
        = descending altitude) or bottom-to-top.  The output is sorted
        in descending order (highest pressure first) per CMIP convention.
    """
    xr = _import_xarray()
    plev_np = np.sort(np.asarray(plev, dtype=np.float64))[::-1]
    return xr.DataArray(
        plev_np,
        dims=("plev",),
        attrs={
            "standard_name": "air_pressure",
            "long_name": "Pressure",
            "units": "Pa",
            "axis": "Z",
            "positive": "down",
        },
    )


def _make_depth_da(depth: np.ndarray, *, kind: str = "soil"):
    """Build a CF-compliant depth DataArray.

    ``kind="soil"`` (default): depth below land surface (used by ``Lmon``
    soil fields like ``tsl``).
    ``kind="ocean"``: depth below sea surface (used by ``Omon`` 3-D ocean
    fields like ``thetao``, ``so``, ``uo``, ``vo``, ``wo``,
    ``rhopoto``).  CMIP6 OMIP uses the same dim name ``depth`` for the
    ocean vertical axis with ``positive="down"``.
    """
    xr = _import_xarray()
    if kind == "ocean":
        long_name = "Ocean Depth"
    elif kind == "soil":
        long_name = "Depth Below Land Surface"
    else:
        raise ValueError(
            f"_make_depth_da: kind must be 'ocean' or 'soil'; got {kind!r}."
        )
    return xr.DataArray(
        np.asarray(depth, dtype=np.float64),
        dims=("depth",),
        attrs={
            "standard_name": "depth",
            "long_name": long_name,
            "units": "m",
            "axis": "Z",
            "positive": "down",
        },
    )


def _make_time_da(
    time_val: float,
    ref_date: str,
    calendar: str,
):
    """Build a scalar time DataArray (days since ref_date)."""
    xr = _import_xarray()
    return xr.DataArray(
        np.array([time_val], dtype=np.float64),
        dims=("time",),
        attrs={
            "standard_name": "time",
            "long_name": "Time",
            "units": f"days since {ref_date}",
            "calendar": calendar,
            "axis": "T",
        },
    )


def _make_time_bounds_da(
    bounds: Tuple[float, float],
    ref_date: str,
    calendar: str,
):
    """Build a time_bnds DataArray, shape (1, 2)."""
    xr = _import_xarray()
    return xr.DataArray(
        np.array([[bounds[0], bounds[1]]], dtype=np.float64),
        dims=("time", "bnds"),
        attrs={
            "units": f"days since {ref_date}",
            "calendar": calendar,
        },
    )


# =========================================================================
# CMIP6 metadata helpers
# =========================================================================

_VARIANT_RE = re.compile(
    r"^r(?P<r>\d+)i(?P<i>\d+)p(?P<p>\d+)f(?P<f>\d+)$"
)


def _parse_variant_label(variant_label: str) -> Tuple[int, int, int, int]:
    """Parse a CMIP6 variant label into its four indices.

    ``"r1i1p1f1"`` → ``(1, 1, 1, 1)`` → ``(realization, initialization,
    physics, forcing)``.  Raises ``ValueError`` on malformed input.
    """
    m = _VARIANT_RE.match(variant_label.strip())
    if m is None:
        raise ValueError(
            f"Malformed CMIP6 variant_label {variant_label!r}; "
            "expected 'r<i>i<i>p<i>f<i>' (e.g. 'r1i1p1f1')."
        )
    return (
        int(m.group("r")),
        int(m.group("i")),
        int(m.group("p")),
        int(m.group("f")),
    )


# Ordered, descending CMIP6 nominal_resolution CV thresholds [km].
# See https://github.com/PCMDI/cmip6-cmor-tables/blob/main/Tables/CMIP6_CV.json
# ``nominal_resolution`` — the smallest bucket whose upper bound is ≥ the
# actual grid spacing.  We follow the CMIP6 spec: pick the CV value that
# best brackets the mean great-circle cell dimension in km.
_NOMINAL_RES_BUCKETS_KM: Tuple[Tuple[float, str], ...] = (
    (0.5, "0.5 km"),
    (1.0, "1 km"),
    (2.5, "2.5 km"),
    (5.0, "5 km"),
    (10.0, "10 km"),
    (25.0, "25 km"),
    (50.0, "50 km"),
    (100.0, "100 km"),
    (250.0, "250 km"),
    (500.0, "500 km"),
    (1000.0, "1000 km"),
    (2500.0, "2500 km"),
    (5000.0, "5000 km"),
    (10000.0, "10000 km"),
)


def _compute_nominal_resolution(
    lat: np.ndarray, lon: np.ndarray,
) -> str:
    """Pick the CMIP6 CV ``nominal_resolution`` string for a lat-lon grid.

    Uses the mean grid spacing in degrees, converted to km at the equator
    (1° ≈ 111.19 km), and rounds up to the nearest CMIP6 CV bucket.
    """
    lat_np = np.asarray(lat, dtype=np.float64)
    lon_np = np.asarray(lon, dtype=np.float64)
    if lat_np.size < 2 or lon_np.size < 2:
        return "unknown"
    dlat = float(np.mean(np.abs(np.diff(lat_np))))
    dlon = float(np.mean(np.abs(np.diff(lon_np))))
    # Equatorial km for the longer side (coarser spacing dominates)
    spacing_deg = max(dlat, dlon)
    spacing_km = spacing_deg * 111.19
    for threshold, label in _NOMINAL_RES_BUCKETS_KM:
        if spacing_km <= threshold:
            return label
    return "10000 km"


def _generate_tracking_id() -> str:
    """Return a CMIP6-style tracking ID (``hdl:21.14100/<uuid>``)."""
    return f"hdl:21.14100/{uuid.uuid4()}"


def _realm_for_table(table_id: str) -> str:
    """Return CMIP6 realm CV value for a CMOR table."""
    return _TABLE_REALM.get(table_id, "atmos")


# =========================================================================
# Global attributes
# =========================================================================

def _global_attrs(
    experiment_id: str,
    model_id: str,
    variant_label: str = "r1i1p1f1",
    grid_label: str = "gn",
    institution: str = "Columbia University",
    institution_id: str = "CU",
    source: str = "legoESM: Differentiable Earth System Model in JAX",
    source_type: str = "AGCM",
    sub_experiment_id: str = "none",
    parent_experiment_id: str = "no parent",
    parent_source_id: str = "no parent",
    parent_variant_label: str = "no parent",
    parent_activity_id: str = "no parent",
    parent_time_units: str = "no parent",
    license_text: str = CMIP6_LICENSE,
    further_info_url: str = "",
    nominal_resolution: str = "unknown",
    tracking_id: str = "",
) -> Dict[str, str]:
    """Return standard CF/CMIP6 global attributes.

    Populates all attributes required by the CMIP6 controlled vocabulary
    (``required_global_attributes`` in the CMIP6_CV.json), so the output
    passes PrePARE / cmip6-cmor-tables metadata validation once the
    ``institution_id`` / ``source_id`` are registered with PCMDI.

    The *parent_** attributes default to ``"no parent"``, which is the
    CV-compliant sentinel for experiments branched from no parent run
    (e.g. ``amip``, ``piControl``).  For branched experiments pass the
    actual parent identifiers.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    try:
        r_idx, i_idx, p_idx, f_idx = _parse_variant_label(variant_label)
    except ValueError:
        r_idx = i_idx = p_idx = f_idx = 1
    return {
        "Conventions": "CF-1.8",
        "mip_era": "CMIP6",
        "activity_id": "CMIP",
        "experiment_id": experiment_id,
        "sub_experiment": "none",
        "sub_experiment_id": sub_experiment_id,
        "institution": institution,
        "institution_id": institution_id,
        "source_id": model_id,
        "source": source,
        "source_type": source_type,
        "product": "model-output",
        "realm": "atmos",
        "variant_label": variant_label,
        "realization_index": np.int32(r_idx),
        "initialization_index": np.int32(i_idx),
        "physics_index": np.int32(p_idx),
        "forcing_index": np.int32(f_idx),
        "grid_label": grid_label,
        "nominal_resolution": nominal_resolution,
        "creation_date": now,
        "tracking_id": tracking_id,
        "license": license_text,
        "further_info_url": further_info_url,
        "parent_experiment_id": parent_experiment_id,
        "parent_source_id": parent_source_id,
        "parent_variant_label": parent_variant_label,
        "parent_activity_id": parent_activity_id,
        "parent_time_units": parent_time_units,
        "branch_method": "no parent",
        "branch_time_in_child": np.float64(0.0),
        "branch_time_in_parent": np.float64(0.0),
        "frequency": "",
        "table_id": "",
        "variable_id": "",
        "history": f"Created by legoESM CFWriter on {now}",
    }


# =========================================================================
# CFWriter class
# =========================================================================

class CFWriter:
    """Manage CF/CMOR-compliant NetCDF output for a legoESM experiment.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory.  Sub-directories are created per table.
    experiment_id : str
        Experiment identifier (e.g. ``"amip"``, ``"piControl"``).
    model_id : str
        Model source identifier (e.g. ``"legoESM-1-0"``).
    freq : str
        Output frequency label: ``"mon"`` or ``"day"``.
    calendar : str
        CF calendar type.  Default ``"noleap"`` (equivalent to
        ``"365_day"``).
    ref_date : str
        Reference date for the time axis, e.g. ``"0001-01-01"``.
    variant_label : str
        CMIP6 variant label.  Default ``"r1i1p1f1"``.
    grid_label : str
        Grid label.  Default ``"gn"`` (native grid).
    institution : str
        Institution string for global attributes.
    compress_level : int
        NetCDF4 deflate compression level (0-9).  Default 4.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        experiment_id: str,
        model_id: str,
        freq: str = "mon",
        calendar: str = "noleap",
        ref_date: str = "1850-01-01",
        variant_label: str = "r1i1p1f1",
        grid_label: str = "gn",
        institution: str = "Columbia University",
        institution_id: str = "CU",
        source_type: str = "AGCM",
        sub_experiment_id: str = "none",
        parent_experiment_id: str = "no parent",
        parent_source_id: str = "no parent",
        parent_variant_label: str = "no parent",
        parent_activity_id: str = "no parent",
        parent_time_units: str = "no parent",
        license_text: str = CMIP6_LICENSE,
        further_info_url: str = "",
        compress_level: int = 4,
    ) -> None:
        # Validate variant_label up front — malformed labels would
        # otherwise silently fall back to (1,1,1,1) for the index
        # attributes, which is a subtle CMIP6 validation failure.
        _parse_variant_label(variant_label)

        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.freq = freq
        self.calendar = calendar
        self.ref_date = ref_date
        self.variant_label = variant_label
        self.grid_label = grid_label
        self.institution = institution
        self.institution_id = institution_id
        self.source_type = source_type
        self.sub_experiment_id = sub_experiment_id
        self.parent_experiment_id = parent_experiment_id
        self.parent_source_id = parent_source_id
        self.parent_variant_label = parent_variant_label
        self.parent_activity_id = parent_activity_id
        self.parent_time_units = parent_time_units
        self.license_text = license_text
        self.further_info_url = further_info_url
        self.compress_level = compress_level

        # Track open datasets for appending
        self._open_datasets: Dict[str, Any] = {}  # var_name -> xr.Dataset
        self._file_paths: Dict[str, Path] = {}    # var_name -> file path

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _base_global_attrs(
        self,
        table_id: str,
        var_name: str,
        nominal_resolution: str = "unknown",
    ) -> Dict[str, Any]:
        """Build global attributes for a specific variable file.

        A fresh ``tracking_id`` (CMIP6-style persistent-handle UUID) is
        generated per file, and ``realm`` is derived from *table_id*.
        """
        attrs = _global_attrs(
            experiment_id=self.experiment_id,
            model_id=self.model_id,
            variant_label=self.variant_label,
            grid_label=self.grid_label,
            institution=self.institution,
            institution_id=self.institution_id,
            source_type=self.source_type,
            sub_experiment_id=self.sub_experiment_id,
            parent_experiment_id=self.parent_experiment_id,
            parent_source_id=self.parent_source_id,
            parent_variant_label=self.parent_variant_label,
            parent_activity_id=self.parent_activity_id,
            parent_time_units=self.parent_time_units,
            license_text=self.license_text,
            further_info_url=self.further_info_url,
            nominal_resolution=nominal_resolution,
            tracking_id=_generate_tracking_id(),
        )
        attrs["realm"] = _realm_for_table(table_id)
        attrs["frequency"] = self.freq
        attrs["table_id"] = table_id
        attrs["variable_id"] = var_name
        # external_variables: areacella for atmos, areacella/sftlf for land,
        # areacello for ocean.  These cell-area files live alongside the
        # variable files in the DRS and are referenced by name.
        realm = attrs["realm"]
        if realm == "atmos":
            attrs["external_variables"] = "areacella"
        elif realm == "land":
            attrs["external_variables"] = "areacella sftlf"
        elif realm == "ocean":
            attrs["external_variables"] = "areacello"
        elif realm == "seaIce":
            attrs["external_variables"] = "areacello"
        return attrs

    def _output_path(
        self,
        var_name: str,
        table_id: str,
        time_range: str = "",
    ) -> Path:
        """Build the DRS-compliant output file path.

        Pattern:
            <output_dir>/<table_id>/
                <var>_<table>_<model>_<expt>_<variant>_<grid>[_<trange>].nc
        """
        table_dir = self.output_dir / table_id
        table_dir.mkdir(parents=True, exist_ok=True)

        parts = [
            var_name,
            table_id,
            self.model_id,
            self.experiment_id,
            self.variant_label,
            self.grid_label,
        ]
        if time_range:
            parts.append(time_range)
        filename = "_".join(parts) + ".nc"
        return table_dir / filename

    def _encoding_for(self, var_name: str) -> Dict[str, Any]:
        """Return NetCDF encoding dict for a variable."""
        output_dtype = _resolve_output_dtype(None)
        enc: Dict[str, Any] = {
            "dtype": output_dtype.str,
        }
        if self.compress_level > 0:
            enc["zlib"] = True
            enc["complevel"] = self.compress_level
        return enc

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def write_field(
        self,
        var_name: str,
        data: Any,
        time: float,
        time_bounds: Tuple[float, float],
        lat: Any,
        lon: Any,
        plev: Optional[Any] = None,
        depth: Optional[Any] = None,
        table: Optional[str] = None,
        extra_attrs: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Write a single field snapshot as a CF-compliant NetCDF file.

        Parameters
        ----------
        var_name : str
            CMOR short variable name (e.g. ``"tas"``).
        data : array-like
            Field data.  Shape must match the variable's declared
            dimensions: ``(nlat, nlon)`` for 2-D surface fields,
            ``(nplev, nlat, nlon)`` for 3-D pressure-level fields, or
            ``(ndepth, nlat, nlon)`` for soil-depth fields.
            JAX arrays are converted to numpy automatically.
        time : float
            Time coordinate value (days since *ref_date*).
        time_bounds : tuple of float
            ``(t_start, t_end)`` for the averaging period.
        lat : array-like
            Latitude values in degrees north, shape ``(nlat,)``.
        lon : array-like
            Longitude values in degrees east, shape ``(nlon,)``.
        plev : array-like, optional
            Pressure levels in Pa.  Required for 3-D atmospheric fields.
        depth : array-like, optional
            Soil depth levels in metres.  Required for ``tsl`` and
            similar soil-depth fields.
        table : str, optional
            Force a CMOR table (``"Amon"`` or ``"Lmon"``).
        extra_attrs : dict, optional
            Additional variable-level attributes to include.

        Returns
        -------
        Path
            Path to the written NetCDF file.

        Notes
        -----
        Cubed-sphere data must be regridded to a regular lat-lon grid
        before calling this method.  Use
        ``legoesm.grids.regridding.apply_regrid_weights()`` or an
        equivalent offline regridding step.
        """
        xr = _import_xarray()
        table_id, entry = lookup_cmor_entry(var_name, table=table)
        output_dtype = _resolve_output_dtype(None)
        data_np = _to_numpy(data).astype(output_dtype)

        # --- Build coordinates ---
        coords: Dict[str, Any] = {}
        dims: List[str] = ["time"]

        time_da = _make_time_da(time, self.ref_date, self.calendar)
        time_bnds_da = _make_time_bounds_da(
            time_bounds, self.ref_date, self.calendar,
        )
        coords["time"] = time_da

        declared_dims = entry["dimensions"]

        if "plev" in declared_dims:
            if plev is None:
                raise ValueError(
                    f"Variable {var_name!r} requires pressure levels "
                    f"(plev), but none were provided."
                )
            coords["plev"] = _make_plev_da(_to_numpy(plev))
            dims.append("plev")

        if "depth" in declared_dims:
            if depth is None:
                raise ValueError(
                    f"Variable {var_name!r} requires depth levels, "
                    f"but none were provided."
                )
            # Explicit per-table depth-kind dispatch: every table that
            # uses a ``depth`` dimension MUST appear here so the
            # coordinate carries the right CF ``long_name``.  An
            # unknown table fails fast rather than silently writing
            # ``Depth Below Land Surface`` for ocean variables.
            _DEPTH_KIND_BY_TABLE = {
                "Omon": "ocean",
                "Lmon": "soil",
                # Add new tables here as variables with ``depth``
                # are introduced.
            }
            try:
                depth_kind = _DEPTH_KIND_BY_TABLE[table_id]
            except KeyError as exc:
                raise ValueError(
                    f"Variable {var_name!r} uses dim 'depth' but its "
                    f"table {table_id!r} is not registered in the "
                    f"depth-kind dispatch.  Add it to _DEPTH_KIND_BY_TABLE "
                    f"in cmor_output.py."
                ) from exc
            coords["depth"] = _make_depth_da(
                _to_numpy(depth), kind=depth_kind,
            )
            dims.append("depth")

        lat_np = _to_numpy(lat)
        lon_np = _to_numpy(lon)
        coords["lat"] = _make_lat_da(lat_np)
        coords["lon"] = _make_lon_da(lon_np)
        dims.extend(["lat", "lon"])

        # CMIP6 scalar reference-height coordinate for surface diagnostics
        # (tas @ 2 m, uas/vas @ 10 m, etc.).  Attach it to the DataArray's
        # own ``coords`` (not the surrounding Dataset) so that xarray's
        # auto-``coordinates``-attribute logic only tags the data variable
        # — tagging ``time_bnds``/``lat_bnds``/``lon_bnds`` with
        # ``coordinates=height`` would be invalid CF.
        ref_height_m = _VAR_REFERENCE_HEIGHT.get(var_name)
        if ref_height_m is not None:
            coords["height"] = _make_height_da(ref_height_m)

        # Add leading time dimension to data
        data_np = np.expand_dims(data_np, axis=0)  # (1, ...)

        # --- Build DataArray ---
        var_attrs = {
            "standard_name": entry["standard_name"],
            "long_name": entry["long_name"],
            "units": entry["units"],
            "cell_methods": entry["cell_methods"],
        }
        if extra_attrs:
            var_attrs.update(extra_attrs)

        da = xr.DataArray(
            data_np,
            dims=tuple(dims),
            coords=coords,
            attrs=var_attrs,
            name=var_name,
        )

        # --- Assemble Dataset ---
        ds = da.to_dataset()
        ds["time_bnds"] = time_bnds_da
        ds["time"].attrs["bounds"] = "time_bnds"
        ds["lat_bnds"] = _make_lat_bnds_da(lat_np)
        ds["lon_bnds"] = _make_lon_bnds_da(lon_np)
        # Suppress xarray's auto ``coordinates`` attribute on bnds vars:
        # scalar Dataset coords like ``height`` would otherwise be
        # written onto every variable, producing invalid CF output on
        # time_bnds/lat_bnds/lon_bnds.
        for _bnds in ("time_bnds", "lat_bnds", "lon_bnds"):
            if _bnds in ds.variables:
                ds[_bnds].encoding["coordinates"] = None
        # NOTE on bnds units: xarray normalizes CF time encoding so that
        # ``time_bnds`` inherits units/calendar from its parent ``time``
        # variable (per CF 1.8) and strips explicit attrs on serialize.
        # Attempts to re-attach via attrs or encoding are no-ops. This
        # is valid CF: modern CDO/ESMValTool accept bnds without units.
        ds.attrs = self._base_global_attrs(
            table_id, var_name,
            nominal_resolution=_compute_nominal_resolution(lat_np, lon_np),
        )

        # --- Write to disk ---
        out_path = self._output_path(var_name, table_id)
        encoding = {
            var_name: self._encoding_for(var_name),
        }
        if out_path.exists():
            # Append by extending the time dimension in place.
            # This avoids reading + concatenating + rewriting the
            # entire file, which is O(n²) over a multi-year run.
            nc4 = _import_netcdf4()
            if nc4 is not None:
                with nc4.Dataset(str(out_path), "a") as ncf:
                    t_idx = len(ncf.dimensions["time"])
                    ncf.variables["time"][t_idx] = float(time)
                    ncf.variables["time_bnds"][t_idx, :] = [
                        time_bounds[0], time_bounds[1],
                    ]
                    ncf.variables[var_name][t_idx] = data_np[0]
            else:
                # Fallback: xarray concat (original O(n²) path)
                existing = xr.open_dataset(out_path, decode_times=False)
                ds = xr.concat([existing, ds], dim="time")
                existing.close()
                ds.to_netcdf(
                    out_path, format="NETCDF4",
                    encoding=encoding, unlimited_dims=["time"],
                )
        else:
            ds.to_netcdf(
                out_path, format="NETCDF4",
                encoding=encoding, unlimited_dims=["time"],
            )

        return out_path

    def write_monthly(
        self,
        monthly_data: Dict[str, Any],
        lat: Any,
        lon: Any,
        plev: Optional[Any] = None,
        depth: Optional[Any] = None,
        start_year: int = 1,
    ) -> List[Path]:
        """Write monthly-mean data from a MonthlyAccumulator to NetCDF.

        This convenience method takes the output of
        ``MonthlyAccumulator.finalize()`` (a dict with keys like
        ``"zonal_tas"``, ``"scalar_pr"``, ``"profile_ta"``, etc.) and
        writes each recognized CMOR variable to its own CF-compliant
        file.

        Parameters
        ----------
        monthly_data : dict
            Output of ``MonthlyAccumulator.finalize()``.  Expected keys:

            - ``"months"`` : list of ``(year, month)`` tuples
            - ``"zonal_<name>"`` : ``(n_months, n_lat)`` — 2-D fields
            - ``"profile_<name>"`` : ``(n_months, n_lat, nlev)`` — 3-D
            - ``"scalar_<name>"`` : ``(n_months,)`` — global means
            - ``"lat"`` : ``(n_lat,)`` — latitude bin centers

        lat : array-like
            1-D latitude for the output grid (degrees north).
        lon : array-like
            1-D longitude for the output grid (degrees east).
        plev : array-like, optional
            Pressure levels in Pa for 3-D fields.
        depth : array-like, optional
            Soil depth levels in metres for soil-profile fields.
        start_year : int
            Year offset for time axis computation (default 1).

        Returns
        -------
        list of Path
            Paths to all files written.

        Notes
        -----
        The MonthlyAccumulator stores zonal means (1-D in latitude).
        This method broadcasts them to ``(nlat, nlon)`` by repeating
        along the longitude axis, which is appropriate for zonal-mean
        diagnostics.  For full 2-D output, the caller should provide
        regridded data in the accumulator.
        """
        xr = _import_xarray()
        months: List[Tuple[int, int]] = monthly_data.get("months", [])
        if not months:
            return []

        lat_np = _to_numpy(lat)
        lon_np = _to_numpy(lon)
        nlat = lat_np.shape[0]
        nlon = lon_np.shape[0]
        n_months = len(months)
        output_dtype = _resolve_output_dtype(None)

        # Days in each month (noleap / 365_day calendar)
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Compute time axis: mid-month in days since ref_date
        time_vals = np.empty(n_months, dtype=np.float64)
        time_bnds = np.empty((n_months, 2), dtype=np.float64)
        for i, (yr, mo) in enumerate(months):
            year_offset = (yr - start_year) * 365.0
            day_start = year_offset + sum(month_days[:mo - 1])
            day_end = day_start + month_days[mo - 1]
            time_vals[i] = 0.5 * (day_start + day_end)
            time_bnds[i, 0] = day_start
            time_bnds[i, 1] = day_end

        written: List[Path] = []
        written_vars: set = set()  # track var names already written

        # Process keys in priority order: zonal > profile > scalar
        # so that spatially-resolved data takes precedence.
        priority = {"zonal_": 0, "profile_": 1, "scalar_": 2}
        sorted_keys = sorted(
            (k for k in monthly_data if k not in ("months", "lat")),
            key=lambda k: next(
                (v for p, v in priority.items() if k.startswith(p)), 9
            ),
        )

        # ---- Pre-fetch all device arrays in a single ``jax.device_get`` ----
        # so the JAX runtime can pipeline the per-variable transfers in
        # parallel.  The previous per-key ``_to_numpy(arr_raw)`` chain
        # forced ~50 sequential device→host blocking syncs at every
        # CMIP6 monthly write.
        _device_values = [monthly_data[k] for k in sorted_keys]
        try:
            _host_values = jax.device_get(_device_values)
        except Exception:
            _host_values = _device_values
        _host_lookup = {k: v for k, v in zip(sorted_keys, _host_values)}

        # ---- Process each key in the monthly_data dict ----
        for key in sorted_keys:
            arr_raw = _host_lookup[key]
            arr = _to_numpy(arr_raw)

            # Determine CMOR variable name from key prefix
            if key.startswith("zonal_"):
                var_name = key[len("zonal_"):]
            elif key.startswith("profile_"):
                var_name = key[len("profile_"):]
            elif key.startswith("scalar_"):
                var_name = key[len("scalar_"):]
            else:
                # Unknown key format — skip
                continue

            # Skip if we already wrote a higher-priority version
            if var_name in written_vars:
                continue

            # Check if this is a recognized CMOR variable
            try:
                table_id, entry = lookup_cmor_entry(var_name)
            except KeyError:
                # Not a standard CMOR variable — skip silently
                continue

            declared_dims = entry["dimensions"]

            # Build the field array in the correct shape
            if key.startswith("zonal_"):
                # arr shape: (n_months, n_lat_bins)
                # Broadcast to (n_months, nlat, nlon)
                if arr.shape[1] != nlat:
                    # Interpolate from accumulator latitude bins to
                    # output latitude
                    acc_lat = _to_numpy(monthly_data.get(
                        "lat",
                        np.linspace(-90, 90, arr.shape[1]),
                    ))
                    from numpy import interp as np_interp
                    new_arr = np.empty((n_months, nlat), dtype=output_dtype)
                    for t in range(n_months):
                        new_arr[t] = np_interp(lat_np, acc_lat, arr[t])
                    arr = new_arr

                # Broadcast along longitude
                field = np.broadcast_to(
                    arr[:, :, np.newaxis],
                    (n_months, nlat, nlon),
                ).copy().astype(output_dtype)

            elif key.startswith("profile_"):
                # arr shape: (n_months, n_lat_bins, nlev)
                if "plev" not in declared_dims:
                    continue

                nplev = arr.shape[2] if arr.ndim == 3 else 0
                if nplev == 0:
                    continue

                # Interpolate latitude if needed
                if arr.shape[1] != nlat:
                    acc_lat = _to_numpy(monthly_data.get(
                        "lat",
                        np.linspace(-90, 90, arr.shape[1]),
                    ))
                    from numpy import interp as np_interp
                    new_arr = np.empty(
                        (n_months, nlat, nplev), dtype=output_dtype,
                    )
                    for t in range(n_months):
                        for k in range(nplev):
                            new_arr[t, :, k] = np_interp(
                                lat_np, acc_lat, arr[t, :, k],
                            )
                    arr = new_arr

                # Broadcast along longitude: (n_months, nplev, nlat, nlon)
                # Rearrange from (n_months, nlat, nplev) -> (n_months, nplev, nlat)
                arr_transposed = np.transpose(arr, (0, 2, 1))
                field = np.broadcast_to(
                    arr_transposed[:, :, :, np.newaxis],
                    (n_months, nplev, nlat, nlon),
                ).copy().astype(output_dtype)

            elif key.startswith("scalar_"):
                # arr shape: (n_months,)
                # Broadcast to (n_months, nlat, nlon) as uniform field
                field = np.broadcast_to(
                    arr[:, np.newaxis, np.newaxis],
                    (n_months, nlat, nlon),
                ).copy().astype(output_dtype)

            else:
                continue

            # ---- Build xarray Dataset ----
            coords: Dict[str, Any] = {}
            dims_list: List[str] = ["time"]

            time_da = xr.DataArray(
                time_vals,
                dims=("time",),
                attrs={
                    "standard_name": "time",
                    "long_name": "Time",
                    "units": f"days since {self.ref_date}",
                    "calendar": self.calendar,
                    "axis": "T",
                    "bounds": "time_bnds",
                },
            )
            coords["time"] = time_da

            if "plev" in declared_dims and key.startswith("profile_"):
                plev_vals = (
                    _to_numpy(plev) if plev is not None
                    else CMIP6_PLEV19[:nplev]
                )
                coords["plev"] = _make_plev_da(plev_vals)
                dims_list.append("plev")

            if "depth" in declared_dims:
                if depth is not None:
                    coords["depth"] = _make_depth_da(_to_numpy(depth))
                    dims_list.append("depth")
                else:
                    continue  # skip depth-requiring vars without depth

            coords["lat"] = _make_lat_da(lat_np)
            coords["lon"] = _make_lon_da(lon_np)
            dims_list.extend(["lat", "lon"])

            # Attach scalar reference height to the DataArray's own coords
            # (not the Dataset via assign_coords) so xarray does not
            # auto-tag time_bnds/lat_bnds/lon_bnds with coordinates="height".
            ref_height_m = _VAR_REFERENCE_HEIGHT.get(var_name)
            if ref_height_m is not None:
                coords["height"] = _make_height_da(ref_height_m)

            var_attrs = {
                "standard_name": entry["standard_name"],
                "long_name": entry["long_name"],
                "units": entry["units"],
                "cell_methods": entry["cell_methods"],
            }

            da = xr.DataArray(
                field,
                dims=tuple(dims_list),
                coords=coords,
                attrs=var_attrs,
                name=var_name,
            )
            ds = da.to_dataset()

            # Time bounds
            ds["time_bnds"] = xr.DataArray(
                time_bnds,
                dims=("time", "bnds"),
                attrs={
                    "units": f"days since {self.ref_date}",
                    "calendar": self.calendar,
                },
            )
            # Spatial cell bounds (required by CF/CMIP6)
            ds["lat_bnds"] = _make_lat_bnds_da(lat_np)
            ds["lon_bnds"] = _make_lon_bnds_da(lon_np)
            # Suppress xarray's auto ``coordinates`` attribute on bnds
            # variables so scalar coords like ``height`` don't produce
            # invalid CF output (coordinates="height" on bnds is wrong).
            for _bnds in ("time_bnds", "lat_bnds", "lon_bnds"):
                if _bnds in ds.variables:
                    ds[_bnds].encoding["coordinates"] = None
            # xarray strips explicit bnds units per CF normalization;
            # see note in ``write_field``.

            ds.attrs = self._base_global_attrs(
                table_id, var_name,
                nominal_resolution=_compute_nominal_resolution(lat_np, lon_np),
            )

            # Time range string for filename
            yr0, mo0 = months[0]
            yr1, mo1 = months[-1]
            time_range = (
                f"{yr0:04d}{mo0:02d}-{yr1:04d}{mo1:02d}"
            )

            out_path = self._output_path(var_name, table_id, time_range)
            encoding = {var_name: self._encoding_for(var_name)}

            ds.to_netcdf(
                out_path,
                format="NETCDF4",
                encoding=encoding,
                unlimited_dims=["time"],
            )
            written.append(out_path)
            written_vars.add(var_name)

        return written

    def write_daily(
        self,
        daily_data: Dict[str, Any],
        lat: Any,
        lon: Any,
    ) -> List[Path]:
        """Write daily-mean data from a ``SpatialDailyAccumulator`` to NetCDF.

        Consumes the output of ``SpatialDailyAccumulator.finalize()``:

        - ``"days"`` : list of ``(year, doy)`` tuples
        - ``"field_2d_<name>"`` : ``(n_days, nlat, nlon)`` — daily mean
        - ``"field_2d_<name>_min"`` / ``"_max"`` : daily extremes

        Each recognized variable is written to the CMIP6 ``day`` table.
        ``tas`` extremes become ``tasmin`` / ``tasmax`` via the standard
        CMIP6 naming convention.

        Parameters
        ----------
        daily_data : dict
            Output of ``SpatialDailyAccumulator.finalize()``.
        lat, lon : array-like
            1-D latitude / longitude of the output grid.

        Returns
        -------
        list of Path
            Unique paths written (one per variable, with all days
            appended as the ``time`` dimension).
        """
        days: List[Tuple[int, int]] = daily_data.get("days", [])
        if not days:
            return []

        # Accumulator key → CMIP6 variable name
        key_to_var: Dict[str, str] = {}
        for k in daily_data:
            if not k.startswith("field_2d_"):
                continue
            inner = k[len("field_2d_"):]
            if inner.endswith("_min"):
                key_to_var[k] = inner[:-len("_min")] + "min"
            elif inner.endswith("_max"):
                key_to_var[k] = inner[:-len("_max")] + "max"
            else:
                key_to_var[k] = inner

        written: List[Path] = []
        seen: set = set()
        for key, var_name in key_to_var.items():
            # Skip variables not defined in the day table.
            try:
                lookup_cmor_entry(var_name, table="day")
            except KeyError:
                continue

            field = _to_numpy(daily_data[key])
            if field.ndim != 3 or field.shape[0] != len(days):
                continue

            out_path: Optional[Path] = None
            for i, (yr, doy) in enumerate(days):
                # ref_date corresponds to year {self._cmip_start_year}.
                # Buckets store year relative to that, so simply:
                #   day-offset = yr * 365 + (doy - 1)   (noleap)
                d0 = float(yr * 365 + (doy - 1))
                out_path = self.write_field(
                    var_name=var_name,
                    data=field[i],
                    time=d0 + 0.5,
                    time_bounds=(d0, d0 + 1.0),
                    lat=lat,
                    lon=lon,
                    table="day",
                )
            if out_path is not None and out_path not in seen:
                written.append(out_path)
                seen.add(out_path)

        return written

    def write_fixed(
        self,
        var_name: str,
        data: Any,
        lat: Any,
        lon: Any,
        extra_attrs: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Write a time-invariant field (``fx`` table) to NetCDF.

        ``fx`` files are written once per experiment and contain no time
        dimension. Typical variables: ``orog`` (surface altitude),
        ``areacella`` (cell area), ``sftlf`` (land fraction).

        Parameters
        ----------
        var_name : str
            Short CMOR variable name; must be in the ``fx`` table.
        data : array-like
            2-D field, shape ``(nlat, nlon)``.
        lat, lon : array-like
            1-D latitude / longitude.
        extra_attrs : dict, optional
            Additional variable attributes.

        Returns
        -------
        Path
            Written file path.
        """
        xr = _import_xarray()
        table_id, entry = lookup_cmor_entry(var_name, table="fx")
        output_dtype = _resolve_output_dtype(None)

        lat_np = _to_numpy(lat)
        lon_np = _to_numpy(lon)
        data_np = _to_numpy(data).astype(output_dtype)
        if data_np.shape != (lat_np.shape[0], lon_np.shape[0]):
            raise ValueError(
                f"write_fixed: expected data shape ({lat_np.shape[0]}, "
                f"{lon_np.shape[0]}), got {data_np.shape}"
            )

        var_attrs = {
            "standard_name": entry["standard_name"],
            "long_name": entry["long_name"],
            "units": entry["units"],
            "cell_methods": entry["cell_methods"],
        }
        if extra_attrs:
            var_attrs.update(extra_attrs)

        da = xr.DataArray(
            data_np,
            dims=("lat", "lon"),
            coords={
                "lat": _make_lat_da(lat_np),
                "lon": _make_lon_da(lon_np),
            },
            attrs=var_attrs,
            name=var_name,
        )
        ds = da.to_dataset()
        ds["lat_bnds"] = _make_lat_bnds_da(lat_np)
        ds["lon_bnds"] = _make_lon_bnds_da(lon_np)
        # Same bnds-coord suppression as in write_field / write_monthly.
        for _bnds in ("lat_bnds", "lon_bnds"):
            if _bnds in ds.variables:
                ds[_bnds].encoding["coordinates"] = None

        ds.attrs = self._base_global_attrs(
            table_id, var_name,
            nominal_resolution=_compute_nominal_resolution(lat_np, lon_np),
        )
        # fx files have no time axis — override frequency.
        ds.attrs["frequency"] = "fx"

        out_path = self._output_path(var_name, table_id)
        encoding = {var_name: self._encoding_for(var_name)}
        ds.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
        return out_path

    def close(self) -> None:
        """Finalize and close any open dataset handles.

        This is a no-op in the current implementation (each
        ``write_field`` / ``write_monthly`` call writes and closes
        immediately), but is provided for forward compatibility with
        buffered writing.
        """
        for ds in self._open_datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        self._open_datasets.clear()
        self._file_paths.clear()

    def __enter__(self) -> "CFWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"CFWriter(output_dir={self.output_dir!r}, "
            f"experiment_id={self.experiment_id!r}, "
            f"model_id={self.model_id!r}, "
            f"freq={self.freq!r})"
        )
