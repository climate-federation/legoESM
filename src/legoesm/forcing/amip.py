"""AMIP forcing: prescribed SST and sea-ice from PCMDI reference datasets.

Loads monthly-mean SST and sea-ice concentration from NetCDF files
(COBE-SST2, HadISST, or custom), regrids to the cubed-sphere grid,
and provides linear time interpolation for use during integration.

Data sources:
- COBE-SST2: ftp://ftp.cgd.ucar.edu/archive/SSTICE/MODEL.SST.COBE-SST2.*.nc
  Variables: SST_cpl [K], ice_cov [%]
- HadISST: Met Office Hadley Centre (1x1 deg monthly)
  Variables: sst [C], sic [fraction]

References
----------
- Taylor, K. E., Williamson, D., & Zwiers, F. (2000). The sea surface
  temperature and sea-ice concentration boundary conditions for AMIP II
  simulations. PCMDI Report No. 60.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class AMIPForcingConfig(NamedTuple):
    """Configuration for AMIP boundary conditions.

    Fields
    ------
    dataset : str
        Dataset preset: "cobe", "hadisst", or "custom".
    path : str
        Path to NetCDF file.
    sst_var : str
        SST variable name in NetCDF.
    sic_var : str
        Sea-ice concentration variable name in NetCDF.
    time_var : str
        Time variable name.
    lat_var : str
        Latitude variable name.
    lon_var : str
        Longitude variable name.
    sst_offset : float
        Additive offset for SST (e.g., +273.15 if data in Celsius) [K].
    sic_scale : float
        Multiplicative scale for SIC (e.g., 0.01 if data in percent).
    T_ice : float
        Sea-ice surface temperature [K].
    albedo_ice : float
        Sea-ice albedo.
    albedo_ocean : float
        Open ocean albedo.
    """
    dataset: str = "cobe"
    path: str = ""
    sst_var: str = "SST_cpl"
    sic_var: str = "ice_cov"
    time_var: str = "time"
    lat_var: str = "lat"
    lon_var: str = "lon"
    sst_offset: float = 0.0
    sic_scale: float = 1.0
    T_ice: float = 271.35
    albedo_ice: float = 0.65
    albedo_ocean: float = 0.06


class AMIPForcing(NamedTuple):
    """Loaded and regridded AMIP forcing data.

    Fields
    ------
    times : jax.Array
        Time coordinate as days since first record, shape (ntime,).
    sst : jax.Array
        Sea surface temperature [K], shape (ntime, ...) where ``...``
        is ``(6, n, n)`` for cubed-sphere or ``(n_lat, n_lon)`` for Gaussian.
    sic : jax.Array
        Sea-ice concentration [0-1], same shape as sst.
    config : AMIPForcingConfig
        Configuration used to load the data.
    """
    times: jnp.ndarray
    sst: jnp.ndarray
    sic: jnp.ndarray
    config: AMIPForcingConfig


def get_amip_preset(dataset_name: str) -> AMIPForcingConfig:
    """Return an AMIPForcingConfig with preset variable names and offsets.

    Parameters
    ----------
    dataset_name : str
        One of "cobe", "hadisst".

    Returns
    -------
    AMIPForcingConfig
        Config with correct variable names and unit conversions.
    """
    if dataset_name == "cobe":
        return AMIPForcingConfig(
            dataset="cobe",
            sst_var="SST_cpl",
            sic_var="ice_cov",
            sst_offset=0.0,       # already in K
            sic_scale=0.01,       # percent -> fraction
        )
    elif dataset_name == "hadisst":
        return AMIPForcingConfig(
            dataset="hadisst",
            sst_var="sst",
            sic_var="sic",
            sst_offset=273.15,    # Celsius -> Kelvin
            sic_scale=1.0,        # already fraction
        )
    else:
        raise ValueError(
            f"Unknown dataset preset: {dataset_name!r}. "
            f"Use 'cobe', 'hadisst', or create a custom AMIPForcingConfig."
        )


def load_amip_forcing(config: AMIPForcingConfig, grid) -> AMIPForcing:
    """Load AMIP forcing from NetCDF and regrid to the target grid.

    Parameters
    ----------
    config : AMIPForcingConfig
        Forcing configuration with file path and variable names.
    grid : CubedSphereGrid or GaussianGrid
        Target grid.  Detected via ``hasattr(grid, 'n_lat')``.

    Returns
    -------
    AMIPForcing
        Regridded forcing data as JAX arrays.
        Shape is (ntime, 6, n, n) for cubed-sphere or
        (ntime, n_lat, n_lon) for Gaussian.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    path = config.path
    if not path:
        raise ValueError("AMIPForcingConfig.path is empty — provide a NetCDF file path")

    ds = xr.open_dataset(path)

    # --- Validate required variables ---
    missing = []
    for vname, label in [
        (config.sst_var, "SST"),
        (config.sic_var, "SIC"),
        (config.lat_var, "latitude"),
        (config.lon_var, "longitude"),
        (config.time_var, "time"),
    ]:
        if vname not in ds:
            missing.append(f"  {label}: expected variable '{vname}'")
    if missing:
        available = ", ".join(sorted(ds.data_vars.keys() | ds.coords.keys()))
        raise KeyError(
            f"Missing variables in {path}:\n"
            + "\n".join(missing)
            + f"\nAvailable: {available}"
        )

    # Extract source grid
    lat_src = ds[config.lat_var].values.astype(np.float64)
    lon_src = ds[config.lon_var].values.astype(np.float64)

    # Ensure longitude is in [0, 360) for wrapping
    lon_src = lon_src % 360.0

    # Sort by longitude if needed
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]

    # Extract SST and SIC
    sst_data = ds[config.sst_var].values  # (ntime, nlat, nlon) or (ntime, nlon, nlat)
    sic_data = ds[config.sic_var].values

    # Handle dimension ordering: ensure (ntime, nlat, nlon)
    # Check if shapes match (lat, lon) order
    if sst_data.ndim == 3:
        # Reorder longitude
        sst_data = sst_data[:, :, lon_order]
        sic_data = sic_data[:, :, lon_order]
    elif sst_data.ndim == 2:
        # Single time step
        sst_data = sst_data[None, :, lon_order]
        sic_data = sic_data[None, :, lon_order]

    # Apply unit conversions
    sst_data = sst_data.astype(np.float64) + config.sst_offset
    sic_data = sic_data.astype(np.float64) * config.sic_scale

    # Fill NaN (land points) with nearest neighbor
    sst_data = _fill_nan_nearest(sst_data, lat_src, lon_src)
    sic_data = _fill_nan_nearest(sic_data, lat_src, lon_src)

    # Clamp SIC to [0, 1]
    sic_data = np.clip(sic_data, 0.0, 1.0)

    # Ensure SST is physically reasonable (at least freezing)
    sst_data = np.maximum(sst_data, 200.0)

    # Wrap longitude for interpolation continuity
    # Pad one column at each end
    lon_wrapped = np.concatenate([lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0])
    sst_wrapped = np.concatenate([sst_data[:, :, -1:], sst_data, sst_data[:, :, :1]], axis=2)
    sic_wrapped = np.concatenate([sic_data[:, :, -1:], sic_data, sic_data[:, :, :1]], axis=2)

    # Detect grid type: Gaussian grids have 'n_lat'; cubed-sphere has 'n'.
    is_gaussian = hasattr(grid, 'n_lat') and not hasattr(grid, 'n')

    if is_gaussian:
        # Gaussian grid: lat is (n_lat,) in radians, lon is (n_lon,) in radians
        target_lat_1d = np.asarray(grid.lat) * 180.0 / np.pi   # (n_lat,) degrees
        target_lon_1d = np.asarray(grid.lon) * 180.0 / np.pi   # (n_lon,) degrees
        target_lon_1d = target_lon_1d % 360.0
        target_lon_2d, target_lat_2d = np.meshgrid(target_lon_1d, target_lat_1d)
        target_shape = (grid.n_lat, grid.n_lon)
    else:
        # Cubed-sphere
        target_lat_2d = np.asarray(grid.lat) * 180.0 / np.pi
        target_lon_2d = np.asarray(grid.lon) * 180.0 / np.pi
        target_lon_2d = target_lon_2d % 360.0
        target_shape = (6, grid.n, grid.n)

    ntime = sst_data.shape[0]
    sst_regridded = np.zeros((ntime, *target_shape), dtype=np.float64)
    sic_regridded = np.zeros((ntime, *target_shape), dtype=np.float64)

    target_points = np.stack([target_lat_2d.ravel(), target_lon_2d.ravel()], axis=-1)

    for t in range(ntime):
        interp_sst = RegularGridInterpolator(
            (lat_src, lon_wrapped), sst_wrapped[t],
            method="linear", bounds_error=False, fill_value=None,
        )
        interp_sic = RegularGridInterpolator(
            (lat_src, lon_wrapped), sic_wrapped[t],
            method="linear", bounds_error=False, fill_value=None,
        )
        sst_regridded[t] = interp_sst(target_points).reshape(target_shape)
        sic_regridded[t] = interp_sic(target_points).reshape(target_shape)

    # Time axis: days since first record
    time_coord = ds[config.time_var].values
    if np.issubdtype(time_coord.dtype, np.datetime64):
        t0 = time_coord[0]
        times_days = (time_coord - t0) / np.timedelta64(1, "D")
        times_days = times_days.astype(np.float64)
    else:
        # Assume already in days or similar numeric
        times_days = time_coord.astype(np.float64)
        times_days = times_days - times_days[0]

    ds.close()

    # Clamp SIC again after interpolation
    sic_regridded = np.clip(sic_regridded, 0.0, 1.0)

    return AMIPForcing(
        times=jnp.array(times_days),
        sst=jnp.array(sst_regridded),
        sic=jnp.array(sic_regridded),
        config=config,
    )


def get_forcing_at_time(
    forcing: AMIPForcing,
    day: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Linearly interpolate SST and SIC to a given day.

    Parameters
    ----------
    forcing : AMIPForcing
        Loaded forcing data.
    day : float
        Day since start of forcing record.

    Returns
    -------
    sst : jax.Array
        Interpolated SST [K], same spatial shape as forcing.sst[0].
    sic : jax.Array
        Interpolated SIC [0-1], same spatial shape as forcing.sic[0].
    """
    times = forcing.times
    ntime = times.shape[0]

    # Clamp day to forcing time range
    day = jnp.clip(day, times[0], times[-1])

    # Find bracketing indices
    idx = jnp.searchsorted(times, day, side="right") - 1
    idx = jnp.clip(idx, 0, ntime - 2)
    idx_next = idx + 1

    # Interpolation weight
    dt = times[idx_next] - times[idx]
    dt = jnp.maximum(dt, 1e-10)  # avoid division by zero
    weight = (day - times[idx]) / dt

    # Linear interpolation
    sst = (1.0 - weight) * forcing.sst[idx] + weight * forcing.sst[idx_next]
    sic = (1.0 - weight) * forcing.sic[idx] + weight * forcing.sic[idx_next]

    # Clamp SIC
    sic = jnp.clip(sic, 0.0, 1.0)

    return sst, sic


def _fill_nan_nearest(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Fill NaN values with nearest valid neighbor on lat-lon grid.

    Parameters
    ----------
    data : np.ndarray
        Data array, shape (ntime, nlat, nlon). NaN marks missing values.
    lat : np.ndarray
        Latitude array [degrees], shape (nlat,).
    lon : np.ndarray
        Longitude array [degrees], shape (nlon,).

    Returns
    -------
    np.ndarray
        Data with NaNs filled.
    """
    if not np.any(np.isnan(data)):
        return data

    from scipy.interpolate import NearestNDInterpolator

    # Build coordinate meshgrid
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Convert to Cartesian for distance (handles wrap-around)
    lat_rad = np.deg2rad(lat_2d)
    lon_rad = np.deg2rad(lon_2d)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    filled = data.copy()
    for t in range(data.shape[0]):
        frame = data[t]
        mask_valid = ~np.isnan(frame.ravel())
        if mask_valid.all() or not mask_valid.any():
            continue

        interp = NearestNDInterpolator(
            coords[mask_valid], frame.ravel()[mask_valid]
        )
        frame_filled = interp(coords).reshape(frame.shape)
        filled[t] = frame_filled

    return filled
