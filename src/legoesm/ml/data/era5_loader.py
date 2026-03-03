"""ERA5 reanalysis data loader for SFNO training.

Loads ERA5 data from Zarr stores (e.g. WeatherBench2 format) and
prepares input-target pairs for autoregressive training.

The loader uses xarray for lazy loading and supports selecting
specific variables, pressure levels, and time ranges.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm.ml.channel_packing import WB2_PRESSURE_LEVELS


class ERA5Config(NamedTuple):
    """Configuration for ERA5 data loading.

    Attributes
    ----------
    zarr_store : str
        Path or URL to the Zarr store containing ERA5 data.
    variables : tuple of str
        Variable names to load (e.g. ("geopotential", "temperature",
        "u_component_of_wind", "v_component_of_wind",
        "specific_humidity")).
    levels : tuple of int
        Pressure levels [hPa] to select.
    time_range : tuple of str
        (start, end) date strings, e.g. ("1979-01-01", "2020-12-31").
    resolution : str
        Spatial resolution label, e.g. "1.40625deg".
    dt_hours : int
        Time step between consecutive samples [hours].
    """
    zarr_store: str = ""
    variables: tuple = (
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    )
    levels: tuple = WB2_PRESSURE_LEVELS
    time_range: tuple = ("1979-01-01", "2020-12-31")
    resolution: str = "1.40625deg"
    dt_hours: int = 6


def create_era5_dataset(config: ERA5Config):
    """Open an ERA5 Zarr store as an xarray.Dataset.

    Parameters
    ----------
    config : ERA5Config
        Data configuration.

    Returns
    -------
    xarray.Dataset
        Lazy-loaded dataset with selected variables, levels, and
        time range.

    Raises
    ------
    ImportError
        If xarray or zarr are not installed.
    FileNotFoundError
        If the Zarr store does not exist.
    """
    import xarray as xr

    ds = xr.open_zarr(config.zarr_store)

    # Select time range
    start, end = config.time_range
    ds = ds.sel(time=slice(start, end))

    # Select pressure levels if present
    if "level" in ds.dims:
        ds = ds.sel(level=list(config.levels))

    # Select variables
    available = [v for v in config.variables if v in ds]
    ds = ds[available]

    return ds


def load_era5_batch(
    config: ERA5Config,
    batch_size: int = 4,
    dt_hours: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load a random batch of (input, target) pairs from ERA5.

    Each sample consists of two consecutive time steps separated by
    ``dt_hours``. The input is the earlier state and the target is
    the later state.

    Parameters
    ----------
    config : ERA5Config
        Data configuration.
    batch_size : int
        Number of samples in the batch.
    dt_hours : int, optional
        Override for the time step between input and target.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    inputs : jax.Array, shape (batch_size, n_lat, n_lon, n_channels)
        Input states.
    targets : jax.Array, shape (batch_size, n_lat, n_lon, n_channels)
        Target states (dt_hours later).
    """
    if dt_hours is None:
        dt_hours = config.dt_hours
    if rng is None:
        rng = np.random.default_rng()

    ds = create_era5_dataset(config)

    # Number of time steps between input and target
    time_stride = dt_hours // config.dt_hours
    n_times = len(ds.time) - time_stride

    # Random time indices
    t_indices = rng.choice(n_times, size=batch_size, replace=False)

    inputs_list = []
    targets_list = []

    for t in t_indices:
        inp = _dataset_to_array(ds.isel(time=int(t)), config)
        tgt = _dataset_to_array(ds.isel(time=int(t + time_stride)), config)
        inputs_list.append(inp)
        targets_list.append(tgt)

    inputs = jnp.stack(inputs_list, axis=0)
    targets = jnp.stack(targets_list, axis=0)

    return inputs, targets


def _dataset_to_array(ds_slice, config: ERA5Config) -> jnp.ndarray:
    """Convert a single time slice to a packed array.

    Stacks all variables and levels into a single array of shape
    (n_lat, n_lon, n_channels).
    """
    arrays = []
    for var in config.variables:
        if var not in ds_slice:
            continue
        data = ds_slice[var].values
        if data.ndim == 2:
            # Surface variable: (lat, lon) → (lat, lon, 1)
            arrays.append(data[..., None])
        elif data.ndim == 3:
            # Pressure-level variable: (level, lat, lon) → (lat, lon, level)
            arrays.append(np.moveaxis(data, 0, -1))

    packed = np.concatenate(arrays, axis=-1)
    return jnp.array(packed, dtype=jnp.float32)
