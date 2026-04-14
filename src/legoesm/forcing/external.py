"""External forcing interfaces for AMIP/CMIP-style experiments.

Provides configuration and interpolation for time-varying external
forcings beyond SST/SIC:

- Greenhouse gas concentrations (CO2, CH4, N2O, CFCs)
- Ozone climatology / forcing
- Aerosol optical depth climatology
- Solar irradiance variations

Status
------
- GHG: config + constant-value active; time-varying from file supported
- Ozone: zonal-mean climatology with lat/vertical interpolation + reference fallback
- Aerosol: zonal-mean AOD climatology with optional volcanic contribution
- Solar: constant/file TSI and full spectral (per g-point) forcing
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Data loading helpers (Zarr-first, NetCDF fallback via xarray)
# ==============================================================================

def _open_forcing_dataset(path: str):
    """Open a forcing file as xarray Dataset (Zarr or NetCDF).

    Always opens with ``decode_times=False`` so that time coordinates are
    returned as raw numeric values (float/int) that can be safely cast to
    ``np.float64``.  Files whose dates extend beyond numpy's datetime64[ns]
    range (~2262) would otherwise cause xarray to fall back to
    ``cftime.datetime`` objects, which cannot be cast to float64.
    """
    import os
    import xarray as xr
    if os.path.isdir(path) or path.endswith(".zarr"):
        return xr.open_zarr(path, decode_times=False)
    return xr.open_dataset(path, decode_times=False)


@lru_cache(maxsize=16)
def _load_timeseries(path: str, varnames: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load 1-D time series variables from a Zarr store or NetCDF file.

    Parameters
    ----------
    path : str
        Path to a Zarr store or NetCDF file with a ``time`` dimension
        (in fractional days).
    varnames : tuple of str
        Variable names to load (must be 1-D along time).

    Returns
    -------
    (times, data) where times is shape (N,) in days and data maps
    each varname to a 1-D numpy array of length N.
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds.dims:
        ds.close()
        raise ValueError(f"Forcing file {path!r} has no 'time' dimension")
    times = np.asarray(ds["time"].values, dtype=np.float64)
    data = {}
    for v in varnames:
        if v not in ds.data_vars:
            ds.close()
            raise ValueError(f"Variable {v!r} not found in {path!r}")
        arr = np.asarray(ds[v].values, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != times.shape[0]:
            ds.close()
            raise ValueError(
                f"Variable {v!r} must be 1-D with length matching 'time' "
                f"(got shape {arr.shape}, expected ({times.shape[0]},))"
            )
        data[v] = arr
    ds.close()
    return times, data


def _interp_1d(times: np.ndarray, values: np.ndarray, day: float) -> float:
    """Linearly interpolate a 1-D time series at *day*, clamping at edges."""
    return float(np.interp(day, times, values))


def _interp_2d_time(times: np.ndarray, values: np.ndarray, day: float) -> np.ndarray:
    """Linearly interpolate a 2-D time series (time, nfeat) at *day*."""
    if values.ndim != 2 or values.shape[0] != times.shape[0]:
        raise ValueError(
            "Expected values with shape (ntime, nfeat) matching times length; "
            f"got {values.shape} with times {times.shape}",
        )
    out = np.zeros((values.shape[1],), dtype=np.float64)
    for j in range(values.shape[1]):
        out[j] = np.interp(day, times, values[:, j])
    return out


@lru_cache(maxsize=16)
def _load_monthly_zonal(path: str, varname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a monthly zonal-mean field from a Zarr store or NetCDF file.

    Handles files with full lat/lon grids (e.g. Kinne aerosol files with
    dims ``(time, band, lat, lon)``) by averaging over lon and any extra
    non-(time, lat) dimensions to produce a ``(ntime, nlat)`` array.

    Returns
    -------
    (mid_days, lat, data) where mid_days is shape (ntime,), lat is shape
    (nlat,), and data is shape (ntime, nlat).
    """
    ds = _open_forcing_dataset(path)
    if varname not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable {varname!r} not found in {path!r}")
    var = ds[varname]
    data = np.asarray(var.values, dtype=np.float64)
    dims = list(var.dims)

    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' variable in {path!r}")
    if "time" in ds:
        mid_days = np.asarray(ds["time"].values, dtype=np.float64)
    else:
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
    ds.close()

    # Average over longitude to produce a zonal mean.
    for lname in ("lon", "longitude"):
        if lname in dims:
            ax = dims.index(lname)
            data = np.nanmean(data, axis=ax)
            dims.pop(ax)
            break

    # Average over any remaining non-(time, lat) dimensions (e.g. spectral
    # bands in Kinne aerosol files: (time, lnwl, lat) → (time, lat)).
    lat_name = "lat" if "lat" in dims else "latitude"
    time_name = "time" if "time" in dims else None
    keep = {lat_name}
    if time_name:
        keep.add(time_name)
    extra_axes = [i for i, d in enumerate(dims) if d not in keep]
    for ax in sorted(extra_axes, reverse=True):
        data = np.nanmean(data, axis=ax)
        dims.pop(ax)

    return mid_days, lat, data


@lru_cache(maxsize=16)
def _load_monthly_zonal_with_levels(path: str, varname: str):
    """Like ``_load_monthly_zonal`` but also returns pressure levels if present.

    Returns
    -------
    (mid_days, lat, plev, data) where plev is shape (nlev,) in [Pa]
    or None if no vertical dimension.
    """
    ds = _open_forcing_dataset(path)
    if varname not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable {varname!r} not found in {path!r}")
    var = ds[varname]
    data = np.asarray(var.values, dtype=np.float64)
    dims = list(var.dims)
    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' variable in {path!r}")
    if "time" in ds:
        mid_days = np.asarray(ds["time"].values, dtype=np.float64)
    else:
        mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])

    plev = None
    for vname in ("plev", "level", "lev"):
        if vname in ds:
            plev = np.asarray(ds[vname].values, dtype=np.float64)
            # Convert hPa → Pa if needed (detect via units attr or magnitude)
            units = ds[vname].attrs.get("units", "")
            if units in ("hPa", "millibar", "mbar", "mb"):
                plev = plev * 100.0
            elif not units and plev.size > 0 and np.max(plev) < 1500.0:
                # Heuristic: surface pressure ~1013 hPa; if max < 1500 assume hPa
                plev = plev * 100.0
            break

    # Average over longitude if present → zonal mean
    lon_names = ("lon", "longitude")
    for lname in lon_names:
        if lname in dims:
            lon_ax = dims.index(lname)
            data = np.nanmean(data, axis=lon_ax)
            dims.pop(lon_ax)
            break

    # Ensure dimension order is (time, lat, plev) for downstream code.
    # CMIP6 files often have (time, plev, lat); swap if needed.
    if plev is not None and data.ndim == 3:
        lat_name = "lat" if "lat" in dims else "latitude"
        plev_name = next((v for v in ("plev", "level", "lev") if v in dims), None)
        if plev_name and lat_name in dims and plev_name in dims:
            lat_ax = dims.index(lat_name)
            plev_ax = dims.index(plev_name)
            if plev_ax < lat_ax:
                # (time, plev, lat) → (time, lat, plev)
                data = np.swapaxes(data, plev_ax, lat_ax)

    ds.close()
    return mid_days, lat, plev, data


@lru_cache(maxsize=16)
def _load_time_gpt(
    path: str,
    tsi_var: str,
    spectral_var: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Load time-varying solar spectral forcing (Zarr or NetCDF).

    Expected variables
    ------------------
    - ``time`` (days)
    - optional ``tsi`` [W/m^2]
    - spectral weights ``spectral_var`` with shape (time, ngpt)
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds:
        ds.close()
        raise ValueError(f"No 'time' variable in spectral solar file {path!r}")
    times = np.asarray(ds["time"].values, dtype=np.float64)
    if spectral_var not in ds.data_vars:
        ds.close()
        raise ValueError(
            f"Spectral variable {spectral_var!r} not found in {path!r}",
        )
    spec = np.asarray(ds[spectral_var].values, dtype=np.float64)
    if spec.ndim != 2 or spec.shape[0] != times.shape[0]:
        ds.close()
        raise ValueError(
            f"Spectral variable {spectral_var!r} must have shape (time, ngpt); "
            f"got {spec.shape}",
        )
    tsi = None
    if tsi_var in ds.data_vars:
        tsi = np.asarray(ds[tsi_var].values, dtype=np.float64)
        if tsi.ndim != 1 or tsi.shape[0] != times.shape[0]:
            ds.close()
            raise ValueError(
                f"TSI variable {tsi_var!r} must have shape (time,); got {tsi.shape}",
            )
    ds.close()
    return times, tsi, spec


def _interp_monthly_cyclic(mid_days: np.ndarray, data: np.ndarray, day: float) -> np.ndarray:
    """Interpolate a monthly-cyclic field (12, ...) to a day of year.

    Uses cyclic linear interpolation with period 365.25 days.
    """
    period = 365.25
    day_mod = day % period
    n = len(mid_days)
    idx_right = np.searchsorted(mid_days % period, day_mod)
    if idx_right >= n:
        idx_right = 0
    idx_left = (idx_right - 1) % n
    d_left = mid_days[idx_left] % period
    d_right = mid_days[idx_right] % period
    span = (d_right - d_left) % period
    if span == 0:
        return data[idx_left]
    w = ((day_mod - d_left) % period) / span
    return (1 - w) * data[idx_left] + w * data[idx_right]


def _interp_zonal_to_grid(lat_src: np.ndarray, field: np.ndarray,
                           lat_grid: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a zonal-mean field to model grid latitudes.

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitudes [degrees].
    field : (nlat_src,) or (nlat_src, nlev)
        Zonal-mean field at source latitudes.
    lat_grid : jax array, any shape
        Model grid latitudes [radians]. Will be converted to degrees.

    Returns
    -------
    jax array with shape (*lat_grid.shape,) or (*lat_grid.shape, nlev)
    """
    lat_deg = np.asarray(jnp.degrees(lat_grid)).ravel()

    if field.ndim == 1:
        result = np.interp(lat_deg, lat_src, field)
        return jnp.array(result).reshape(lat_grid.shape)
    elif field.ndim == 2:
        nlev = field.shape[1]
        result = np.zeros((len(lat_deg), nlev))
        for k in range(nlev):
            result[:, k] = np.interp(lat_deg, lat_src, field[:, k])
        return jnp.array(result).reshape((*lat_grid.shape, nlev))
    else:
        raise ValueError(f"Expected 1D or 2D field, got {field.ndim}D")


def _interp_vertical(field_plev: jnp.ndarray, plev_src: np.ndarray,
                      p_target: jnp.ndarray) -> jnp.ndarray:
    """Interpolate vertically in log-pressure space.

    Parameters
    ----------
    field_plev : (..., nlev_src)
        Field on source pressure levels.
    plev_src : (nlev_src,)
        Source pressure levels [Pa].
    p_target : (..., nlev_target)
        Target pressure levels [Pa].

    Returns
    -------
    (..., nlev_target) interpolated field.
    """
    log_p_src = np.log(np.maximum(plev_src, 1e-10))
    log_p_tgt = jnp.log(jnp.maximum(p_target, 1e-10))

    # Use numpy interp on flattened arrays
    shape_prefix = field_plev.shape[:-1]
    nlev_tgt = p_target.shape[-1]
    field_flat = np.asarray(field_plev).reshape(-1, len(plev_src))
    log_p_tgt_flat = np.asarray(log_p_tgt).reshape(-1, nlev_tgt)

    result = np.zeros((field_flat.shape[0], nlev_tgt))
    for i in range(field_flat.shape[0]):
        result[i] = np.interp(log_p_tgt_flat[i], log_p_src, field_flat[i])

    return jnp.array(result).reshape((*shape_prefix, nlev_tgt))


# ==============================================================================
# Greenhouse gases
# ==============================================================================

class GHGConfig(NamedTuple):
    """Greenhouse gas configuration.

    When source="constant", the specified concentrations are used at
    all times. When source="file", values are linearly interpolated
    from a NetCDF time series containing co2_ppmv, ch4_ppbv, n2o_ppbv.
    When source="annual_file", values are loaded from a CMIP6-style
    annual global-mean file (e.g. greenhouse_historical_plus.nc) with
    variables CO2, CH4, N2O, CFC_11, CFC_12 dimensioned (time, lat, lon)
    where lat=1, lon=1 and time is in fractional years.

    The concentrations are consumed by RRTMGP radiation (not gray).

    Fields
    ------
    co2_ppmv : float
        CO2 concentration [ppmv]. Default: 348 (AMIP II reference ~1979-1996).
    ch4_ppbv : float
        CH4 concentration [ppbv].
    n2o_ppbv : float
        N2O concentration [ppbv].
    cfc11_pptv : float
        CFC-11 concentration [pptv].
    cfc12_pptv : float
        CFC-12 concentration [pptv].
    source : str
        "constant", "file" (1-D time series), or "annual_file" (CMIP6
        annual global-mean file with year-indexed time axis).
    path : str
        Path to time-varying GHG file (only used if source="file" or
        "annual_file").
    start_year : int
        Simulation start year, used to convert simulation day → calendar
        year when source="annual_file".
    """
    co2_ppmv: float = 348.0
    ch4_ppbv: float = 1650.0
    n2o_ppbv: float = 306.0
    cfc11_pptv: float = 240.0
    cfc12_pptv: float = 530.0
    source: str = "constant"
    path: str = ""
    start_year: int = 1979


@lru_cache(maxsize=4)
def _load_ghg_annual_file(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load CMIP6-style annual global-mean GHG file (Zarr or NetCDF).

    Expected format: variables CO2, CH4, N2O, CFC_11, CFC_12 with
    dimensions (time, lat, lon) where lat=1, lon=1.  The time axis
    uses units ``"year as %Y.%f"`` giving fractional year values
    (e.g. 1850.0, 1851.0, ...).

    Returns
    -------
    (years, data) where years is shape (N,) and data maps variable
    names to 1-D numpy arrays of length N.  Values are in the file's
    native units (CO2 in 1e-6, CH4/N2O in 1e-9, CFCs in 1e-12).
    """
    ds = _open_forcing_dataset(path)
    if "time" not in ds.dims:
        ds.close()
        raise ValueError(f"GHG file {path!r} has no 'time' dimension")
    years = np.asarray(ds["time"].values, dtype=np.float64)

    data = {}
    for varname in ("CO2", "CH4", "N2O", "CFC_11", "CFC_12"):
        if varname not in ds.data_vars:
            ds.close()
            raise ValueError(f"Variable {varname!r} not found in {path!r}")
        arr = np.asarray(ds[varname].values, dtype=np.float64)
        # Squeeze spatial dimensions (lat=1, lon=1) → 1-D time series
        arr = arr.squeeze()
        if arr.ndim != 1 or arr.shape[0] != years.shape[0]:
            ds.close()
            raise ValueError(
                f"Variable {varname!r} must reduce to 1-D after "
                f"squeezing (got shape {arr.shape}, expected "
                f"({years.shape[0]},))"
            )
        data[varname] = arr

    ds.close()
    return years, data


def get_ghg_at_time(config: GHGConfig, day: float) -> dict:
    """Return GHG concentrations at a given simulation day.

    Parameters
    ----------
    config : GHGConfig
    day : float
        Simulation day (unused for constant source).

    Returns
    -------
    dict with keys "co2_ppmv", "ch4_ppbv", "n2o_ppbv", "cfc11_pptv",
    "cfc12_pptv".
    """
    if config.source == "constant":
        return {
            "co2_ppmv": config.co2_ppmv,
            "ch4_ppbv": config.ch4_ppbv,
            "n2o_ppbv": config.n2o_ppbv,
            "cfc11_pptv": config.cfc11_pptv,
            "cfc12_pptv": config.cfc12_pptv,
        }
    elif config.source == "file":
        if not config.path:
            raise ValueError("GHGConfig.path must be set when source='file'")
        varnames = ("co2_ppmv", "ch4_ppbv", "n2o_ppbv")
        times, data = _load_timeseries(config.path, varnames)
        result = {v: _interp_1d(times, data[v], day) for v in varnames}
        result["cfc11_pptv"] = config.cfc11_pptv
        result["cfc12_pptv"] = config.cfc12_pptv
        return result
    elif config.source == "annual_file":
        if not config.path:
            raise ValueError(
                "GHGConfig.path must be set when source='annual_file'"
            )
        years, data = _load_ghg_annual_file(config.path)
        # Convert simulation day → fractional year
        year = config.start_year + day / 365.25
        # File stores mole fractions scaled by unit metadata:
        # CO2 in 1e-6 (ppmv), CH4/N2O in 1e-9 (ppbv), CFCs in 1e-12 (pptv)
        return {
            "co2_ppmv": _interp_1d(years, data["CO2"], year),
            "ch4_ppbv": _interp_1d(years, data["CH4"], year),
            "n2o_ppbv": _interp_1d(years, data["N2O"], year),
            "cfc11_pptv": _interp_1d(years, data["CFC_11"], year),
            "cfc12_pptv": _interp_1d(years, data["CFC_12"], year),
        }
    else:
        raise ValueError(f"Unknown GHG source: {config.source!r}")


def ghg_concentrations_to_vmr(ghg: dict) -> dict:
    """Convert GHG concentrations dict to volume mixing ratios.

    Parameters
    ----------
    ghg : dict
        As returned by ``get_ghg_at_time`` (ppmv/ppbv/pptv units).

    Returns
    -------
    dict mapping RRTMGP gas names to VMR (dimensionless mole fractions).
    """
    vmr = {
        "co2": ghg["co2_ppmv"] * 1.0e-6,
        "ch4": ghg["ch4_ppbv"] * 1.0e-9,
        "n2o": ghg["n2o_ppbv"] * 1.0e-9,
    }
    if "cfc11_pptv" in ghg:
        vmr["cfc11"] = ghg["cfc11_pptv"] * 1.0e-12
    if "cfc12_pptv" in ghg:
        vmr["cfc12"] = ghg["cfc12_pptv"] * 1.0e-12
    return vmr


# ==============================================================================
# Ozone
# ==============================================================================

class OzoneConfig(NamedTuple):
    """Ozone forcing configuration.

    When enabled, provides column ozone or 3D ozone mixing ratio
    for radiation. Supports zonal-mean monthly climatology from
    NetCDF with latitude and optional vertical interpolation to the
    model grid.

    Expected NetCDF schema for climatology:
    - Variable: 'ozone', 'vmro3', 'o3', 'O3', or 'tro3' (auto-detected)
      with dims (time=12, lat[, plev])
    - Units: volume mixing ratio [mol/mol]
    - 'lat' in degrees, 'plev' in Pa (if 3D)

    Fields
    ------
    enabled : bool
        Whether ozone forcing is active.
    source : str
        "climatology" (monthly zonal-mean) or "file" (full 3D).
    path : str
        Path to ozone NetCDF file.
    use_reference_if_missing : bool
        If True and path is empty, use a built-in reference ozone profile.
    reference_p_peak_hPa : float
        Peak pressure for the reference profile [hPa].
    reference_o3_max_vmr : float
        Peak ozone VMR [mol/mol] for the reference profile.
    reference_sigma_logp : float
        Width of reference ozone in log-pressure space.
    reference_lat_dependence : bool
        If True, reference ozone increases toward poles.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""
    use_reference_if_missing: bool = False
    reference_p_peak_hPa: float = 30.0
    reference_o3_max_vmr: float = 8.0e-6
    reference_sigma_logp: float = 1.5
    reference_lat_dependence: bool = True


def _reference_ozone_profile(
    lat_grid: jnp.ndarray,
    p_grid: jnp.ndarray | None,
    config: OzoneConfig,
) -> jnp.ndarray:
    """Reference ozone profile used when external ozone is active without a file."""
    if p_grid is None:
        base = jnp.full(lat_grid.shape, config.reference_o3_max_vmr)
        if config.reference_lat_dependence:
            base = base * (1.0 + 0.5 * jnp.sin(lat_grid) ** 2)
        return jnp.clip(base, 1.0e-10, None)

    p_hPa = p_grid / 100.0
    sigma = config.reference_sigma_logp
    p_peak = config.reference_p_peak_hPa
    o3 = config.reference_o3_max_vmr * jnp.exp(
        -0.5 * ((jnp.log(jnp.maximum(p_hPa, 1.0e-12)) - jnp.log(p_peak)) / sigma) ** 2,
    )
    if config.reference_lat_dependence:
        o3 = o3 * (1.0 + 0.5 * jnp.sin(lat_grid)[..., None] ** 2)
    return jnp.clip(o3, 1.0e-10, None)


_OZONE_VARNAMES = ("ozone", "vmro3", "o3", "O3", "tro3")


@lru_cache(maxsize=16)
def _detect_ozone_varname(path: str) -> str:
    """Auto-detect the ozone variable name in a Zarr store or NetCDF file.

    Tries common names: 'ozone', 'vmro3' (CMIP6), 'o3', 'O3', 'tro3'.
    """
    ds = _open_forcing_dataset(path)
    for name in _OZONE_VARNAMES:
        if name in ds.data_vars:
            ds.close()
            return name
    ds.close()
    raise ValueError(
        f"No ozone variable found in {path!r}. "
        f"Expected one of {_OZONE_VARNAMES}"
    )


def get_ozone_at_time(config: OzoneConfig, day: float,
                       lat_grid: jnp.ndarray | None = None,
                       p_grid: jnp.ndarray | None = None):
    """Return ozone field at a given simulation day.

    Parameters
    ----------
    config : OzoneConfig
    day : float
        Day of year (fractional).
    lat_grid : jax array or None
        Model grid latitudes [radians]. If provided, interpolates
        the zonal-mean climatology to these latitudes.
    p_grid : jax array or None
        Model pressure levels [Pa]. If provided and the ozone file
        contains a vertical dimension, interpolates vertically.

    Returns
    -------
    None if ozone is disabled.
    jax array interpolated to (lat_grid.shape,) or
    (lat_grid.shape, nlev) if 3D, or dict with "lat"/"ozone" keys
    if lat_grid is not provided.
    """
    if not config.enabled:
        return None
    if (not config.path) and config.use_reference_if_missing:
        if lat_grid is None:
            lat_ref = np.linspace(-90.0, 90.0, 181)
            lat_ref_rad = jnp.radians(jnp.array(lat_ref))
            return {"lat": lat_ref, "ozone": np.asarray(_reference_ozone_profile(lat_ref_rad, None, config))}
        return _reference_ozone_profile(lat_grid, p_grid, config)
    if not config.path:
        raise ValueError(
            "OzoneConfig.path must be set when enabled=True unless "
            "use_reference_if_missing=True",
        )

    varname = _detect_ozone_varname(config.path)
    mid_days, lat, plev, data = _load_monthly_zonal_with_levels(
        config.path, varname
    )
    ozone_interp = _interp_monthly_cyclic(mid_days, data, day)

    if lat_grid is not None:
        # Interpolate to model grid latitudes
        ozone_on_grid = _interp_zonal_to_grid(lat, ozone_interp, lat_grid)
        # Vertical interpolation if 3D data and pressure grid provided
        if plev is not None and p_grid is not None and ozone_on_grid.ndim > len(lat_grid.shape):
            ozone_on_grid = _interp_vertical(ozone_on_grid, plev, p_grid)
        return ozone_on_grid
    else:
        return {"lat": lat, "ozone": ozone_interp}


# ==============================================================================
# Aerosols
# ==============================================================================

class AerosolConfig(NamedTuple):
    """Aerosol forcing configuration.

    When enabled, provides prescribed aerosol optical depth from a
    zonal-mean monthly climatology. The AOD modifies SW radiation.

    Expected NetCDF schema for climatology:
    - Variable: 'aod' with dims (time=12, lat)
    - Units: dimensionless optical depth at 550 nm
    - 'lat' in degrees

    Fields
    ------
    enabled : bool
        Whether aerosol forcing is active.
    source : str
        "climatology" or "file".
    path : str
        Path to aerosol NetCDF file.
    use_reference_if_missing : bool
        If True and path is empty, use a built-in reference aerosol field.
    reference_aod_550 : float
        Reference background AOD at 550 nm.
    reference_lat_factor : float
        Strength of low-latitude enhancement in the reference AOD profile.
    volcanic_enabled : bool
        If True, add volcanic aerosol forcing on top of baseline aerosol.
    volcanic_path : str
        NetCDF path for volcanic AOD climatology/time-series (same schema as aerosol).
    volcanic_scale : float
        Multiplicative scaling applied to volcanic AOD.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""
    use_reference_if_missing: bool = False
    reference_aod_550: float = 0.03
    reference_lat_factor: float = 0.35
    volcanic_enabled: bool = False
    volcanic_path: str = ""
    volcanic_scale: float = 1.0


def _reference_aerosol_profile(lat_grid: jnp.ndarray, config: AerosolConfig) -> jnp.ndarray:
    """Reference zonal aerosol optical depth profile."""
    aod = config.reference_aod_550 * (
        1.0 + config.reference_lat_factor * jnp.cos(lat_grid) ** 2
    )
    return jnp.clip(aod, 0.0, None)


def get_aerosol_at_time(config: AerosolConfig, day: float,
                         lat_grid: jnp.ndarray | None = None):
    """Return aerosol optical depth at a given simulation day.

    Parameters
    ----------
    config : AerosolConfig
    day : float
        Day of year (fractional).
    lat_grid : jax array or None
        If provided, interpolates AOD to model grid latitudes.

    Returns
    -------
    None if aerosol is disabled.
    jax array if lat_grid provided, else dict with "lat"/"aod".
    """
    if not config.enabled:
        return None
    if config.path:
        mid_days, lat, data = _load_monthly_zonal(config.path, "aod")
        aod_interp = _interp_monthly_cyclic(mid_days, data, day)
        if lat_grid is not None:
            base_aod = _interp_zonal_to_grid(lat, aod_interp, lat_grid)
        else:
            base_aod = {"lat": lat, "aod": aod_interp}
    elif config.use_reference_if_missing:
        if lat_grid is None:
            lat = np.linspace(-90.0, 90.0, 181)
            lat_rad = jnp.radians(jnp.array(lat))
            base_aod = {"lat": lat, "aod": np.asarray(_reference_aerosol_profile(lat_rad, config))}
        else:
            base_aod = _reference_aerosol_profile(lat_grid, config)
    else:
        raise ValueError(
            "AerosolConfig.path must be set when enabled=True unless "
            "use_reference_if_missing=True",
        )

    # Optional volcanic contribution.
    if config.volcanic_enabled:
        if config.volcanic_path:
            mid_days_v, lat_v, data_v = _load_monthly_zonal(config.volcanic_path, "aod")
            aod_v = _interp_monthly_cyclic(mid_days_v, data_v, day) * config.volcanic_scale
            if lat_grid is not None:
                volc = _interp_zonal_to_grid(lat_v, aod_v, lat_grid)
            else:
                volc = {"lat": lat_v, "aod": aod_v}
        elif config.use_reference_if_missing:
            volc = (
                jnp.zeros_like(base_aod)
                if lat_grid is not None
                else {"lat": base_aod["lat"], "aod": np.zeros_like(base_aod["aod"])}
            )
        else:
            raise ValueError(
                "AerosolConfig.volcanic_path must be set when volcanic_enabled=True "
                "unless use_reference_if_missing=True",
            )

        if lat_grid is not None:
            return jnp.clip(base_aod + volc, 0.0, None)
        volc_aod = volc["aod"]
        if not np.array_equal(base_aod["lat"], volc["lat"]):
            volc_aod = np.interp(base_aod["lat"], volc["lat"], volc_aod)
        return {"lat": base_aod["lat"], "aod": np.clip(base_aod["aod"] + volc_aod, 0.0, None)}

    return base_aod


# ==============================================================================
# Solar irradiance
# ==============================================================================

class SolarConfig(NamedTuple):
    """Solar irradiance configuration.

    The seasonal cycle is handled by the radiation scheme's insolation
    calculation. This config adds support for TSI variations (e.g.,
    solar cycle, volcanic dimming).

    Constant TSI is active by default. Time-varying TSI from file is
    supported via source="file".

    Fields
    ------
    S_0 : float
        Total solar irradiance [W/m2]. Default: 1360.
    source : str
        "constant", "file", or "spectral_file".
    path : str
        Path to forcing NetCDF file.
    tsi_var : str
        Variable name for TSI time series.
    spectral_var : str
        Variable name for per-g-point solar fractions when source="spectral_file".
    normalize_spectral : bool
        If True, normalize interpolated spectral fractions to sum to one.
    start_year : int
        Simulation start year.  Used to convert simulation day (days since
        start) to the absolute day count used in file-based solar records
        whose time axis is expressed as "days since 1850-01-01".
    """
    S_0: float = 1360.0
    source: str = "constant"
    path: str = ""
    tsi_var: str = "tsi"
    spectral_var: str = "solar_fraction_by_gpt"
    normalize_spectral: bool = True
    start_year: int = 1979


def get_solar_forcing_at_time(config: SolarConfig, day: float) -> dict:
    """Return solar forcing dict with broadband TSI and optional spectral weights.

    Returns
    -------
    dict
        Keys:
        - ``"tsi"``: float [W/m^2]
        - ``"solar_fraction_by_gpt"``: jax array (ngpt,) or None
    """
    if config.source == "constant":
        return {"tsi": float(config.S_0), "solar_fraction_by_gpt": None}

    # CMIP6 solar files store time as "days since 1850-01-01".
    # Convert simulation day (days since start_year-01-01) to the same reference.
    _REF_YEAR = 1850
    abs_day = (config.start_year - _REF_YEAR) * 365.25 + day

    if config.source == "file":
        if not config.path:
            raise ValueError("SolarConfig.path must be set when source='file'")
        times, data = _load_timeseries(config.path, (config.tsi_var,))
        return {"tsi": _interp_1d(times, data[config.tsi_var], abs_day), "solar_fraction_by_gpt": None}

    if config.source == "spectral_file":
        if not config.path:
            raise ValueError("SolarConfig.path must be set when source='spectral_file'")
        times, tsi_series, spec_series = _load_time_gpt(
            config.path,
            config.tsi_var,
            config.spectral_var,
        )
        tsi_val = _interp_1d(times, tsi_series, abs_day) if tsi_series is not None else float(config.S_0)
        spec = _interp_2d_time(times, spec_series, abs_day)
        spec = np.clip(spec, 0.0, None)
        if config.normalize_spectral:
            denom = float(np.sum(spec))
            if denom <= 0.0:
                raise ValueError(
                    f"Interpolated spectral forcing has non-positive sum at day={day}",
                )
            spec = spec / denom
        return {"tsi": tsi_val, "solar_fraction_by_gpt": jnp.array(spec)}

    raise ValueError(f"Unknown solar source: {config.source!r}")


def get_tsi_at_time(config: SolarConfig, day: float) -> float:
    """Return total solar irradiance at a given simulation day.

    Parameters
    ----------
    config : SolarConfig
    day : float

    Returns
    -------
    float : TSI [W/m2]
    """
    return float(get_solar_forcing_at_time(config, day)["tsi"])


# ==============================================================================
# Combined external forcing config
# ==============================================================================

class ExternalForcingConfig(NamedTuple):
    """Combined configuration for all external forcings.

    Pass this to the AMIP experiment to enable/configure external forcings.
    All sub-configs default to inactive/constant, so the baseline AMIP
    experiment works without any external forcing files.
    """
    ghg: GHGConfig = GHGConfig()
    ozone: OzoneConfig = OzoneConfig()
    aerosol: AerosolConfig = AerosolConfig()
    solar: SolarConfig = SolarConfig()
