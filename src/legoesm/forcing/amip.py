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

from legoesm import constants


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
    sic_path : str
        Separate SIC file path.  When non-empty, SIC is loaded from
        this file instead of the main ``path``.
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
    sic_path: str = ""
    # NOTE: ``T_ice`` is the seawater freezing point used as the SST
    # floor / SIC ramp threshold — NOT the sea-ice surface temperature.
    # The legacy name is preserved for AMIP config compatibility, but
    # the value tracks ``constants.T_freeze_ocean`` (CLAUDE.md naming
    # discipline gap; renaming is tracked tech debt).
    T_ice: float = 271.35           # = constants.T_freeze_ocean
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
            sst_offset=constants.T_freeze,    # Celsius -> Kelvin
            sic_scale=1.0,        # already fraction
        )
    else:
        raise ValueError(
            f"Unknown dataset preset: {dataset_name!r}. "
            f"Use 'cobe', 'hadisst', or create a custom AMIPForcingConfig."
        )


def _is_icon_unstructured(ds) -> bool:
    """Detect whether a dataset uses the ICON unstructured grid format.

    ICON boundary condition files have a ``cell`` dimension with
    ``clon``/``clat`` coordinates in radians.
    """
    return "cell" in ds.dims and "clon" in ds.coords and "clat" in ds.coords


def _time_coord_to_days(time_coord) -> np.ndarray:
    """Convert numeric, NumPy datetime, or cftime coordinates to relative days."""
    time_arr = np.asarray(time_coord)

    if np.issubdtype(time_arr.dtype, np.datetime64):
        t0 = time_arr[0]
        return ((time_arr - t0) / np.timedelta64(1, "D")).astype(np.float64)

    first = time_arr[0]
    if hasattr(first, "calendar"):
        import cftime

        units = (
            "days since "
            f"{first.year:04d}-{first.month:02d}-{first.day:02d} "
            f"{first.hour:02d}:{first.minute:02d}:{first.second:02d}"
        )
        return np.asarray(
            cftime.date2num(list(time_arr), units=units, calendar=first.calendar),
            dtype=np.float64,
        )

    try:
        values = time_arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Unsupported AMIP time coordinate type. Expected numeric, "
            "numpy.datetime64, or cftime datetimes."
        ) from exc

    return values - values[0]


def _load_icon_unstructured(config: AMIPForcingConfig, grid) -> AMIPForcing:
    """Load AMIP forcing from ICON unstructured NetCDF files.

    Handles separate SST/SIC files (``config.sic_path``), unit
    conversions, and KD-tree nearest-neighbour regridding from the
    ICON cell centroids to the target grid.
    """
    import xarray as xr
    from scipy.spatial import cKDTree

    # --- Open SST file ---
    ds_sst = xr.open_dataset(config.path)
    try:
        clon = ds_sst["clon"].values.astype(np.float64)   # radians
        clat = ds_sst["clat"].values.astype(np.float64)   # radians
        sst_data = ds_sst[config.sst_var].values.astype(np.float64)
        time_coord = ds_sst[config.time_var].values
    finally:
        ds_sst.close()

    # --- Open SIC file (separate or same) ---
    sic_path = config.sic_path or config.path
    ds_sic = xr.open_dataset(sic_path)
    try:
        sic_data = ds_sic[config.sic_var].values.astype(np.float64)
    finally:
        ds_sic.close()

    # Ensure 2D: (ntime, ncells)
    if sst_data.ndim == 1:
        sst_data = sst_data[None, :]
        sic_data = sic_data[None, :]

    # --- Unit conversions ---
    sst_data = sst_data + config.sst_offset
    sic_data = sic_data * config.sic_scale
    sic_data = np.clip(sic_data, 0.0, 1.0)
    # Clamp SST to the seawater freezing point (271.35 K).  Was 200 K
    # — 71 K below physical freezing — which silently allowed
    # unphysical sub-freezing SST values to leak into surface-flux
    # bulk formulas.
    sst_data = np.maximum(sst_data, constants.T_freeze_ocean)

    # --- Build KD-tree from ICON cell centroids (3D Cartesian) ---
    x_src = np.cos(clat) * np.cos(clon)
    y_src = np.cos(clat) * np.sin(clon)
    z_src = np.sin(clat)
    tree = cKDTree(np.stack([x_src, y_src, z_src], axis=-1))

    # --- Target grid points (3D Cartesian) ---
    grid_lat = np.asarray(grid.grid_lat)  # radians
    grid_lon = np.asarray(grid.grid_lon)  # radians
    target_shape = grid_lat.shape
    x_tgt = np.cos(grid_lat.ravel()) * np.cos(grid_lon.ravel())
    y_tgt = np.cos(grid_lat.ravel()) * np.sin(grid_lon.ravel())
    z_tgt = np.sin(grid_lat.ravel())
    _, idx = tree.query(np.stack([x_tgt, y_tgt, z_tgt], axis=-1))

    # --- Vectorised regridding (all timesteps at once) ---
    ntime = sst_data.shape[0]
    sst_regridded = sst_data[:, idx].reshape(ntime, *target_shape)
    sic_regridded = sic_data[:, idx].reshape(ntime, *target_shape)

    # --- Time axis ---
    times_days = _time_coord_to_days(time_coord)

    sic_regridded = np.clip(sic_regridded, 0.0, 1.0)

    from legoesm.core.precision import get_policy
    _dtype = get_policy().storage
    return AMIPForcing(
        times=jnp.array(times_days),
        sst=jnp.array(sst_regridded, dtype=_dtype),
        sic=jnp.array(sic_regridded, dtype=_dtype),
        config=config,
    )


def load_amip_forcing(config: AMIPForcingConfig, grid) -> AMIPForcing:
    """Load AMIP forcing from NetCDF and regrid to the target grid.

    Supports regular lat-lon grids (COBE-SST2, HadISST, custom) and
    ICON unstructured grids (detected via ``cell`` dimension with
    ``clon``/``clat`` coordinates).

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

    try:
        ds_sst = xr.open_dataset(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"AMIP forcing file not found: {path}"
        )
    except Exception as exc:
        raise OSError(
            f"Failed to open AMIP forcing file: {path}\n  {exc}"
        ) from exc

    sic_path = config.sic_path or path
    ds_sic = ds_sst
    if sic_path != path:
        try:
            ds_sic = xr.open_dataset(sic_path)
        except FileNotFoundError:
            ds_sst.close()
            raise FileNotFoundError(
                f"AMIP sea-ice forcing file not found: {sic_path}"
            )
        except Exception as exc:
            ds_sst.close()
            raise OSError(
                f"Failed to open AMIP sea-ice forcing file: {sic_path}\n  {exc}"
            ) from exc

    # --- Detect ICON unstructured grid ---
    if _is_icon_unstructured(ds_sst):
        if ds_sic is not ds_sst:
            ds_sic.close()
        ds_sst.close()
        return _load_icon_unstructured(config, grid)

    try:
        # --- Validate required variables ---
        missing = []
        for vname, label in [
            (config.sst_var, "SST"),
            (config.lat_var, "latitude"),
            (config.lon_var, "longitude"),
            (config.time_var, "time"),
        ]:
            if vname not in ds_sst:
                missing.append(f"  {label}: expected variable '{vname}' in {path}")
        if config.sic_var not in ds_sic:
            missing.append(
                f"  SIC: expected variable '{config.sic_var}' in {sic_path}"
            )
        if missing:
            available = ", ".join(
                sorted(
                    (set(ds_sst.data_vars) | set(ds_sst.coords))
                    | (set(ds_sic.data_vars) | set(ds_sic.coords))
                )
            )
            raise KeyError(
                "Missing variables in AMIP forcing:\n"
                + "\n".join(missing)
                + f"\nAvailable: {available}"
            )

        # Extract source grid from the SST file
        lat_src = ds_sst[config.lat_var].values.astype(np.float64)
        lon_src = ds_sst[config.lon_var].values.astype(np.float64)
        time_coord = ds_sst[config.time_var].values

        # Ensure longitude is in [0, 360) for wrapping
        lon_src = lon_src % 360.0

        # Sort by longitude if needed
        lon_order = np.argsort(lon_src)
        lon_src = lon_src[lon_order]

        # Extract SST and SIC
        sst_data = ds_sst[config.sst_var].values
        sic_data = ds_sic[config.sic_var].values

        if sst_data.ndim == 2:
            sst_data = sst_data[None, ...]
        if sic_data.ndim == 2:
            sic_data = sic_data[None, ...]
        if sst_data.ndim != 3 or sic_data.ndim != 3:
            raise ValueError(
                "AMIP forcing variables must be 2D or 3D with optional time axis. "
                f"Got SST ndim={sst_data.ndim}, SIC ndim={sic_data.ndim}."
            )

        if sic_data.shape[0] != sst_data.shape[0]:
            raise ValueError(
                "SST and SIC forcing must have the same number of time records. "
                f"Got SST={sst_data.shape[0]} and SIC={sic_data.shape[0]}."
            )

        sst_data = sst_data[:, :, lon_order]
        sic_data = sic_data[:, :, lon_order]

        # Apply unit conversions
        sst_data = sst_data.astype(np.float64) + config.sst_offset
        sic_data = sic_data.astype(np.float64) * config.sic_scale

        # Fill NaN (land points) with nearest neighbor
        sst_data = _fill_nan_nearest(sst_data, lat_src, lon_src)
        sic_data = _fill_nan_nearest(sic_data, lat_src, lon_src)

        # Clamp SIC to [0, 1]
        sic_data = np.clip(sic_data, 0.0, 1.0)

        # Ensure SST is at least the seawater freezing point.  Was
        # 200 K — 71 K below physical freezing — which silently
        # allowed unphysical SST values to flow into bulk formulas.
        sst_data = np.maximum(sst_data, constants.T_freeze_ocean)

        # Wrap longitude for interpolation continuity
        # Pad one column at each end
        lon_wrapped = np.concatenate([lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0])
        sst_wrapped = np.concatenate([sst_data[:, :, -1:], sst_data, sst_data[:, :, :1]], axis=2)
        sic_wrapped = np.concatenate([sic_data[:, :, -1:], sic_data, sic_data[:, :, :1]], axis=2)

        # Use protocol: grid_lat gives 2D (or 3D for CS) lat in radians
        grid_lat = np.asarray(grid.grid_lat)
        grid_lon = np.asarray(grid.grid_lon)
        is_gaussian = grid_lat.ndim == 2 and not hasattr(grid, 'n')

        if is_gaussian:
            target_lat_1d = np.asarray(grid.lat) * 180.0 / np.pi
            target_lon_1d = np.asarray(grid.lon) * 180.0 / np.pi
            target_lon_1d = target_lon_1d % 360.0
            target_lon_2d, target_lat_2d = np.meshgrid(target_lon_1d, target_lat_1d)
            target_shape = grid_lat.shape
        else:
            target_lat_2d = grid_lat * 180.0 / np.pi
            target_lon_2d = grid_lon * 180.0 / np.pi
            target_lon_2d = target_lon_2d % 360.0
            target_shape = grid_lat.shape

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
        times_days = _time_coord_to_days(time_coord)
    finally:
        if ds_sic is not ds_sst:
            ds_sic.close()
        ds_sst.close()

    # Clamp SIC again after interpolation
    sic_regridded = np.clip(sic_regridded, 0.0, 1.0)

    from legoesm.core.precision import get_policy
    _dtype = get_policy().storage
    return AMIPForcing(
        times=jnp.array(times_days),
        sst=jnp.array(sst_regridded, dtype=_dtype),
        sic=jnp.array(sic_regridded, dtype=_dtype),
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

    # Wrap day cyclically so multi-year runs repeat the annual cycle
    # instead of clamping at the last record.
    period = times[-1] - times[0]
    day = jnp.where(period > 0, times[0] + (day - times[0]) % period, day)

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
