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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


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
    if hasattr(arr, "to_py"):
        # Older JAX
        return np.asarray(arr.to_py())
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
    "tos": {
        "standard_name": "sea_surface_temperature",
        "long_name": "Sea Surface Temperature",
        "units": "K",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
    "sic": {
        "standard_name": "sea_ice_area_fraction",
        "long_name": "Sea-Ice Area Percentage (Ocean Grid)",
        "units": "%",
        "cell_methods": "time: mean area: mean where sea",
        "dimensions": ("time", "lat", "lon"),
    },
}

_ADAY_VARIABLES: Dict[str, Dict[str, str]] = {
    "tas": {
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "units": "K",
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
}

# Combined lookup for convenience
CMOR_TABLES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Amon": _AMON_VARIABLES,
    "Lmon": _LMON_VARIABLES,
    "Omon": _OMON_VARIABLES,
    "Aday": _ADAY_VARIABLES,
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


def _make_depth_da(depth: np.ndarray):
    """Build a CF-compliant soil depth DataArray."""
    xr = _import_xarray()
    return xr.DataArray(
        np.asarray(depth, dtype=np.float64),
        dims=("depth",),
        attrs={
            "standard_name": "depth",
            "long_name": "Depth Below Land Surface",
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
) -> Dict[str, str]:
    """Return standard CF/CMIP6 global attributes."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "Conventions": "CF-1.8",
        "activity_id": "CMIP",
        "experiment_id": experiment_id,
        "institution": institution,
        "institution_id": institution_id,
        "source_id": model_id,
        "source": source,
        "variant_label": variant_label,
        "grid_label": grid_label,
        "nominal_resolution": "unknown",
        "creation_date": now,
        "tracking_id": "",
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
        ref_date: str = "0001-01-01",
        variant_label: str = "r1i1p1f1",
        grid_label: str = "gn",
        institution: str = "Columbia University",
        compress_level: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.freq = freq
        self.calendar = calendar
        self.ref_date = ref_date
        self.variant_label = variant_label
        self.grid_label = grid_label
        self.institution = institution
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
    ) -> Dict[str, str]:
        """Build global attributes for a specific variable file."""
        attrs = _global_attrs(
            experiment_id=self.experiment_id,
            model_id=self.model_id,
            variant_label=self.variant_label,
            grid_label=self.grid_label,
            institution=self.institution,
        )
        attrs["frequency"] = self.freq
        attrs["table_id"] = table_id
        attrs["variable_id"] = var_name
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
            coords["depth"] = _make_depth_da(_to_numpy(depth))
            dims.append("depth")

        coords["lat"] = _make_lat_da(_to_numpy(lat))
        coords["lon"] = _make_lon_da(_to_numpy(lon))
        dims.extend(["lat", "lon"])

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
        ds.attrs = self._base_global_attrs(table_id, var_name)

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

        # ---- Process each key in the monthly_data dict ----
        for key in sorted_keys:
            arr_raw = monthly_data[key]
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

            ds.attrs = self._base_global_attrs(table_id, var_name)

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
