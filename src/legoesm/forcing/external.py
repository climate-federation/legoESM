"""External forcing interfaces for AMIP/CMIP-style experiments.

Provides scaffolded configuration and interpolation for time-varying
external forcings beyond SST/SIC:

- Greenhouse gas concentrations (CO2, CH4, N2O, CFCs)
- Ozone climatology / forcing
- Aerosol optical depth climatology
- Solar irradiance variations

Status
------
- GHG: config + constant-value mode active; time-varying from file is scaffolded
- Ozone: fully scaffolded (placeholder)
- Aerosol: fully scaffolded (placeholder)
- Solar: seasonal cycle already active in gray radiation; TSI variation scaffolded

These interfaces are designed so that physics parameterizations can query
them without knowing whether the forcing is constant, prescribed from
a file, or computed interactively.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


# ==============================================================================
# NetCDF time-interpolation helper
# ==============================================================================

@lru_cache(maxsize=16)
def _load_nc_timeseries(path: str, varnames: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load 1-D time series variables from a NetCDF file.

    Parameters
    ----------
    path : str
        Path to a NetCDF file with a ``time`` dimension (in fractional days).
    varnames : tuple of str
        Variable names to load (must be 1-D along time).

    Returns
    -------
    (times, data) where times is shape (N,) in days and data maps
    each varname to a 1-D numpy array of length N.
    """
    import netCDF4  # deferred to avoid hard dep at import time

    with netCDF4.Dataset(path, "r") as ds:
        if "time" not in ds.dimensions:
            raise ValueError(f"NetCDF file {path!r} has no 'time' dimension")
        times = np.asarray(ds.variables["time"][:], dtype=np.float64)
        data = {}
        for v in varnames:
            if v not in ds.variables:
                raise ValueError(f"Variable {v!r} not found in {path!r}")
            arr = np.asarray(ds.variables[v][:], dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != times.shape[0]:
                raise ValueError(
                    f"Variable {v!r} must be 1-D with length matching 'time' "
                    f"(got shape {arr.shape}, expected ({times.shape[0]},))"
                )
            data[v] = arr
    return times, data


def _interp_1d(times: np.ndarray, values: np.ndarray, day: float) -> float:
    """Linearly interpolate a 1-D time series at *day*, clamping at edges."""
    return float(np.interp(day, times, values))


@lru_cache(maxsize=16)
def _load_nc_monthly_zonal(path: str, varname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a monthly zonal-mean field from NetCDF.

    Expected dimensions: ``(time=12, lat, [level])``.

    Returns
    -------
    (mid_days, lat, data) where mid_days is shape (12,) giving
    mid-month days, lat is shape (nlat,), and data is shape
    (12, nlat) or (12, nlat, nlev).
    """
    import netCDF4

    with netCDF4.Dataset(path, "r") as ds:
        if varname not in ds.variables:
            raise ValueError(f"Variable {varname!r} not found in {path!r}")
        data = np.asarray(ds.variables[varname][:], dtype=np.float64)
        if "lat" in ds.variables:
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        elif "latitude" in ds.variables:
            lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        else:
            raise ValueError(f"No 'lat'/'latitude' variable in {path!r}")
        if "time" in ds.variables:
            mid_days = np.asarray(ds.variables["time"][:], dtype=np.float64)
        else:
            # Assume 12 months, mid-month day of year
            mid_days = np.array([15.5 + 30.4375 * m for m in range(12)])
    return mid_days, lat, data


def _interp_monthly_cyclic(mid_days: np.ndarray, data: np.ndarray, day: float) -> np.ndarray:
    """Interpolate a monthly-cyclic field (12, ...) to a day of year.

    Uses cyclic linear interpolation with period 365.25 days.
    """
    period = 365.25
    day_mod = day % period
    # Wrap mid_days to ensure cyclic interpolation
    n = len(mid_days)
    # Find bracketing months
    idx_right = np.searchsorted(mid_days % period, day_mod)
    if idx_right >= n:
        idx_right = 0
    idx_left = (idx_right - 1) % n
    d_left = mid_days[idx_left] % period
    d_right = mid_days[idx_right] % period
    # Handle wrap-around
    span = (d_right - d_left) % period
    if span == 0:
        return data[idx_left]
    w = ((day_mod - d_left) % period) / span
    return (1 - w) * data[idx_left] + w * data[idx_right]


# ==============================================================================
# Greenhouse gases
# ==============================================================================

class GHGConfig(NamedTuple):
    """Greenhouse gas configuration.

    Currently used to inform radiation (e.g., scaling LW optical depth).
    The gray radiation does not use GHG concentrations directly, but this
    config provides the interface for future multi-band or correlated-k
    radiation schemes.

    Fields
    ------
    co2_ppmv : float
        CO2 concentration [ppmv]. Default: 348 (AMIP II reference ~1979-1996).
    ch4_ppbv : float
        CH4 concentration [ppbv].
    n2o_ppbv : float
        N2O concentration [ppbv].
    source : str
        "constant" (use values above) or "file" (load from NetCDF).
    path : str
        Path to time-varying GHG file (only used if source="file").
    """
    co2_ppmv: float = 348.0
    ch4_ppbv: float = 1650.0
    n2o_ppbv: float = 306.0
    source: str = "constant"
    path: str = ""


def get_ghg_at_time(config: GHGConfig, day: float) -> dict:
    """Return GHG concentrations at a given simulation day.

    Parameters
    ----------
    config : GHGConfig
    day : float
        Simulation day (unused for constant source).

    Returns
    -------
    dict with keys "co2_ppmv", "ch4_ppbv", "n2o_ppbv"
    """
    if config.source == "constant":
        return {
            "co2_ppmv": config.co2_ppmv,
            "ch4_ppbv": config.ch4_ppbv,
            "n2o_ppbv": config.n2o_ppbv,
        }
    elif config.source == "file":
        if not config.path:
            raise ValueError("GHGConfig.path must be set when source='file'")
        varnames = ("co2_ppmv", "ch4_ppbv", "n2o_ppbv")
        times, data = _load_nc_timeseries(config.path, varnames)
        return {v: _interp_1d(times, data[v], day) for v in varnames}
    else:
        raise ValueError(f"Unknown GHG source: {config.source!r}")


# ==============================================================================
# Ozone
# ==============================================================================

class OzoneConfig(NamedTuple):
    """Ozone forcing configuration.

    Placeholder for prescribed ozone climatology or forcing.
    When active, provides column ozone or 3D ozone mixing ratio
    for radiation calculations.

    Status: fully scaffolded. Not yet connected to radiation.

    Fields
    ------
    enabled : bool
        Whether ozone forcing is active.
    source : str
        "climatology" (monthly zonal-mean) or "file" (full 3D).
    path : str
        Path to ozone NetCDF file.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""


def get_ozone_at_time(config: OzoneConfig, day: float, grid=None, sigma=None):
    """Return ozone field at a given simulation day.

    Parameters
    ----------
    config : OzoneConfig
    day : float
        Day of year (fractional).
    grid : optional
        Grid object (unused for zonal-mean climatology).
    sigma : optional
        Sigma levels (unused for zonal-mean climatology).

    Returns
    -------
    None if ozone is disabled.
    dict with keys ``"lat"``, ``"ozone"`` if enabled.
    ``"ozone"`` has shape ``(nlat,)`` or ``(nlat, nlev)`` interpolated
    to the given day from a monthly climatology.
    """
    if not config.enabled:
        return None
    if not config.path:
        raise ValueError("OzoneConfig.path must be set when enabled=True")
    mid_days, lat, data = _load_nc_monthly_zonal(config.path, "ozone")
    ozone_interp = _interp_monthly_cyclic(mid_days, data, day)
    return {"lat": lat, "ozone": ozone_interp}


# ==============================================================================
# Aerosols
# ==============================================================================

class AerosolConfig(NamedTuple):
    """Aerosol forcing configuration.

    Placeholder for prescribed aerosol optical depth climatology.
    When active, modifies SW radiation via column AOD.

    Status: fully scaffolded. Not yet connected to radiation.

    Fields
    ------
    enabled : bool
        Whether aerosol forcing is active.
    source : str
        "climatology" or "file".
    path : str
        Path to aerosol NetCDF file.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""


def get_aerosol_at_time(config: AerosolConfig, day: float, grid=None):
    """Return aerosol optical depth at a given simulation day.

    Parameters
    ----------
    config : AerosolConfig
    day : float
        Day of year (fractional).
    grid : optional
        Grid object (unused for zonal-mean climatology).

    Returns
    -------
    None if aerosol is disabled.
    dict with keys ``"lat"``, ``"aod"`` if enabled.
    ``"aod"`` has shape ``(nlat,)`` interpolated to the given day from
    a monthly climatology.
    """
    if not config.enabled:
        return None
    if not config.path:
        raise ValueError("AerosolConfig.path must be set when enabled=True")
    mid_days, lat, data = _load_nc_monthly_zonal(config.path, "aod")
    aod_interp = _interp_monthly_cyclic(mid_days, data, day)
    return {"lat": lat, "aod": aod_interp}


# ==============================================================================
# Solar irradiance
# ==============================================================================

class SolarConfig(NamedTuple):
    """Solar irradiance configuration.

    The seasonal cycle is already handled by the gray radiation's
    daily_mean_insolation(). This config adds support for TSI variations
    (e.g., solar cycle, volcanic dimming).

    Status: constant TSI active; time-varying TSI scaffolded.

    Fields
    ------
    S_0 : float
        Total solar irradiance [W/m2]. Default: 1360.
    source : str
        "constant" or "file" (time-varying TSI from NetCDF).
    path : str
        Path to TSI time series file.
    """
    S_0: float = 1360.0
    source: str = "constant"
    path: str = ""


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
    if config.source == "constant":
        return config.S_0
    elif config.source == "file":
        if not config.path:
            raise ValueError("SolarConfig.path must be set when source='file'")
        times, data = _load_nc_timeseries(config.path, ("tsi",))
        return _interp_1d(times, data["tsi"], day)
    else:
        raise ValueError(f"Unknown solar source: {config.source!r}")


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
