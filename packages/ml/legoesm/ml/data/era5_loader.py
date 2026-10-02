"""ERA5 reanalysis data loader for SFNO training.

Loads ERA5 data from Zarr stores on Google Cloud Storage (WeatherBench2 format)
and prepares input-target pairs for autoregressive training.

GCS access is anonymous (public bucket) — no credentials needed.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm.ml.channel_packing import WB2_PRESSURE_LEVELS

# Default WB2 GCS paths
WB2_ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
WB2_CLIMATOLOGY_ZARR = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_1440x721.zarr"
)
WB2_HRES_ZARR = (
    "gs://weatherbench2/datasets/hres/"
    "2016-2022-0012-1440x721.zarr"
)

# WB2 variable naming conventions — maps common short names to WB2 names
_WB2_VAR_ALIASES = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

# Dimension naming conventions — WB2 uses "latitude"/"longitude"
_DIM_ALIASES = {
    "lat": "latitude",
    "lon": "longitude",
}


class ERA5Config(NamedTuple):
    """Configuration for ERA5 data loading.

    Attributes
    ----------
    zarr_store : str
        Path or GCS URL to the Zarr store containing ERA5 data.
    variables : tuple of str
        Variable names to load (WB2 naming convention).
    levels : tuple of int
        Pressure levels [hPa] to select.
    time_range : tuple of str
        (start, end) date strings, e.g. ("1979-01-01", "2020-12-31").
    dt_hours : int
        Time step between consecutive samples [hours].
    """

    zarr_store: str = WB2_ERA5_ZARR
    variables: tuple = (
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    )
    levels: tuple = WB2_PRESSURE_LEVELS
    time_range: tuple = ("1979-01-01", "2020-12-31")
    dt_hours: int = 6


class ERA5ClimatologyConfig(NamedTuple):
    """Configuration for ERA5 climatology loading (for ACC computation).

    Attributes
    ----------
    zarr_store : str
        Path or GCS URL to the climatology Zarr store.
    variables : tuple of str
        Variable names to load.
    levels : tuple of int
        Pressure levels [hPa] to select.
    """

    zarr_store: str = WB2_CLIMATOLOGY_ZARR
    variables: tuple = (
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    )
    levels: tuple = WB2_PRESSURE_LEVELS


def _open_gcs_zarr(zarr_path: str, **kwargs):
    """Open a Zarr store, using gcsfs for GCS paths.

    Parameters
    ----------
    zarr_path : str
        Local path or ``gs://...`` URL.
    **kwargs
        Extra arguments passed to ``xr.open_zarr``.

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr

    if zarr_path.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(zarr_path)
        return xr.open_zarr(store, chunks=None, **kwargs)
    return xr.open_zarr(zarr_path, chunks=None, **kwargs)


def _normalize_dims(ds):
    """Rename WB2 dimension names to standard lat/lon if present."""
    rename = {}
    for short, long in _DIM_ALIASES.items():
        if long in ds.dims and short not in ds.dims:
            rename[long] = short
    if rename:
        ds = ds.rename(rename)
    return ds


def _resolve_variables(ds, variables: tuple[str, ...]) -> list[str]:
    """Resolve variable names, trying aliases for missing variables.

    Raises ``ValueError`` listing any requested variable that cannot be
    resolved — silently dropping it (the previous behaviour) yields a packed
    array with fewer channels than the model was built for, surfacing as a
    confusing shape mismatch far from the cause.
    """
    resolved = []
    unresolved = []
    for var in variables:
        if var in ds:
            resolved.append(var)
        elif var in _WB2_VAR_ALIASES and _WB2_VAR_ALIASES[var] in ds:
            resolved.append(_WB2_VAR_ALIASES[var])
        else:
            # Reverse lookup (long name → short alias).
            match = next(
                (short for short, long in _WB2_VAR_ALIASES.items()
                 if var == long and short in ds),
                None,
            )
            if match is not None:
                resolved.append(match)
            else:
                unresolved.append(var)
    if unresolved:
        raise ValueError(
            f"ERA5 variable(s) not found in dataset (after alias resolution): "
            f"{unresolved}. Available data_vars: {sorted(ds.data_vars)}."
        )
    return resolved


def create_era5_dataset(config: ERA5Config):
    """Open an ERA5 Zarr store as an xarray.Dataset.

    Streams lazily from GCS when given a ``gs://`` path — no local
    download required.

    Parameters
    ----------
    config : ERA5Config
        Data configuration.

    Returns
    -------
    xarray.Dataset
        Lazy-loaded dataset with selected variables, levels, and time range.
    """
    ds = _open_gcs_zarr(config.zarr_store)
    ds = _normalize_dims(ds)

    # Select time range
    start, end = config.time_range
    ds = ds.sel(time=slice(start, end))

    # Select pressure levels if present
    level_dim = "level" if "level" in ds.dims else "pressure_level"
    if level_dim in ds.dims:
        ds = ds.sel({level_dim: list(config.levels)})

    # Select variables
    available = _resolve_variables(ds, config.variables)
    ds = ds[available]

    return ds


def create_climatology_dataset(config: ERA5ClimatologyConfig = ERA5ClimatologyConfig()):
    """Open the ERA5 climatology Zarr store for ACC computation.

    Parameters
    ----------
    config : ERA5ClimatologyConfig
        Climatology configuration.

    Returns
    -------
    xarray.Dataset
        Climatology dataset with selected variables and levels.
    """
    ds = _open_gcs_zarr(config.zarr_store)
    ds = _normalize_dims(ds)

    # Select pressure levels if present
    level_dim = "level" if "level" in ds.dims else "pressure_level"
    if level_dim in ds.dims:
        ds = ds.sel({level_dim: list(config.levels)})

    available = _resolve_variables(ds, config.variables)
    ds = ds[available]

    return ds


def _dataset_to_array(ds_slice, config: ERA5Config) -> jnp.ndarray:
    """Convert a single time slice to a packed array.

    Stacks all variables and levels into a single array of shape
    (n_lat, n_lon, n_channels).

    Handles WB2 naming conventions:
    - Pressure-level vars may use ``level`` or ``pressure_level`` dim
    - Dimension order may be (level, lat, lon) or (lat, lon, level)
    """
    arrays = []
    available_vars = _resolve_variables(ds_slice, config.variables)

    for var in available_vars:
        if var not in ds_slice:
            continue
        data = ds_slice[var].values
        if data.ndim == 2:
            # Surface variable: (lat, lon) -> (lat, lon, 1)
            arrays.append(data[..., None])
        elif data.ndim == 3:
            # Pressure-level variable: determine axis order
            dims = list(ds_slice[var].dims)
            # Find the level axis (not lat/lon)
            spatial = {"lat", "lon", "latitude", "longitude"}
            level_axis = next(
                (i for i, d in enumerate(dims) if d not in spatial), 0
            )
            if level_axis == 0:
                # (level, lat, lon) -> (lat, lon, level)
                arrays.append(np.moveaxis(data, 0, -1))
            elif level_axis == 2:
                # Already (lat, lon, level)
                arrays.append(data)
            else:
                # (lat, level, lon) -> (lat, lon, level)
                arrays.append(np.moveaxis(data, level_axis, -1))

    packed = np.concatenate(arrays, axis=-1)
    return jnp.array(packed, dtype=jnp.float32)
