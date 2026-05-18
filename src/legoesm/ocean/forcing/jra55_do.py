"""JRA55-do reanalysis loader for OMIP-2-style forced ocean runs.

Reference
---------
Tsujino, H., et al. (2020). "JRA-55 based surface dataset for driving
ocean-sea-ice models (JRA55-do)", Ocean Modelling 130, 79-139.

Tsujino, H., et al. (2020). "Evaluation of global ocean-sea-ice
model simulations based on the experimental protocols of the Ocean
Model Intercomparison Project phase 2 (OMIP-2)", Geosci. Model Dev.
13, 3643-3708.

Spec
----
JRA55-do v1.5 provides 3-hourly atmospheric forcing on a 0.5625-deg
grid from 1958 to present. The seven OMIP-2 forcing channels carried
by this loader are:

* ``u10``, ``v10``      [m/s]       3-hourly 10-m wind components
* ``T_air``             [K]         3-hourly 2-m air temperature
* ``q_air``             [kg/kg]     3-hourly 2-m specific humidity
* ``sw_down``           [W/m^2]     3-hourly downward shortwave
* ``lw_down``           [W/m^2]     3-hourly downward longwave
* ``precip``            [kg/m^2/s]  3-hourly precipitation flux
* ``runoff``            [kg/m^2/s]  daily continental runoff flux

The loader caches each calendar year as a zarr store under
``$LEGOESM_CACHE/forcing/jra55_do/<year>.zarr`` (resolved via
``legoesm.ocean.fidelity.cache.sub("forcing")`` to match the existing
pattern). When the cache is missing the loader returns a deterministic
synthetic climatology so the matrix smoke runs need no external data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np


# JRA55-do native grid: 0.5625 deg ~ 640 lon x 320 lat.
JRA55_NLON: int = 640
JRA55_NLAT: int = 320
JRA55_FREQ_HOURS: int = 3       # 3-hourly atmospheric channels


class OceanForcing(NamedTuple):
    """Seven-channel ocean atmospheric forcing on a regular lat-lon grid.

    Arrays have shape ``(n_time, n_lat, n_lon)`` with units documented
    on each field.
    """
    lon: np.ndarray            # (n_lon,) deg E in [0, 360)
    lat: np.ndarray            # (n_lat,) deg N in [-90, 90]
    time_s: np.ndarray         # (n_time,) seconds since the year start
    u10: np.ndarray            # m/s
    v10: np.ndarray            # m/s
    T_air: np.ndarray          # K
    q_air: np.ndarray          # kg/kg
    sw_down: np.ndarray        # W/m^2
    lw_down: np.ndarray        # W/m^2
    precip: np.ndarray         # kg/m^2/s
    runoff: np.ndarray         # kg/m^2/s


def _cache_dir() -> Path:
    from legoesm.ocean.fidelity import cache as _cache
    return _cache.sub("forcing") / "jra55_do"


def synthetic_ocean_forcing(year: int, *,
                             n_time: int = 12,
                             nlon: int = JRA55_NLON,
                             nlat: int = JRA55_NLAT) -> OceanForcing:
    """Deterministic monthly climatology with physically-reasonable bounds.

    Used as the fallback when the on-disk JRA55-do cache is missing.
    Channel ranges match the OMIP-2 evaluation paper's diagnostics:

    * |u10|, |v10| -- 10 m/s zonal mean, sin(2 lat) meridional belt
    * T_air        -- 263 K poles, 300 K equator, +/- 5 K seasonal cycle
    * q_air        -- saturation at T_air * 80 % RH
    * sw_down      -- 200 cos(lat) * (1 + sin(2 pi t)) W/m^2
    * lw_down      -- 300 W/m^2 globally uniform
    * precip       -- 3e-5 kg/m^2/s tropical band, zero in deserts
    * runoff       -- zero (synthetic mode, no continental mask)
    """
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    lat = np.linspace(-90.0, 90.0, nlat, dtype=np.float64)
    time_s = np.linspace(
        0.0, 365.0 * 86400.0, n_time, endpoint=False, dtype=np.float64,
    )

    LAT_R = np.radians(lat)[None, :, None]
    LON_R = np.radians(lon)[None, None, :]
    cos_lat = np.cos(LAT_R)
    sin_2lat = np.sin(2.0 * LAT_R)
    t_year = (time_s / (365.0 * 86400.0))[:, None, None]
    seasonal = np.sin(2.0 * np.pi * t_year)

    out_shape = (n_time, nlat, nlon)
    u10 = np.broadcast_to(10.0 * sin_2lat * np.ones_like(LON_R), out_shape).copy()
    v10 = np.broadcast_to(2.0 * np.sin(LAT_R) * np.cos(LON_R), out_shape).copy()
    T_air = np.broadcast_to(
        283.0 + 17.0 * cos_lat - 5.0 * seasonal + 0.0 * LON_R,
        out_shape,
    ).copy()
    # Bolton (1980) saturation: e_s = 6.112 exp(17.67 T_C / (T_C+243.5)) hPa
    T_C = T_air - 273.15
    e_s = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))   # hPa
    p_sfc = 1013.25                                      # hPa
    q_sat = 0.622 * e_s / (p_sfc - 0.378 * e_s)
    q_air = 0.8 * q_sat                                  # 80 % RH

    sw_down = np.maximum(
        200.0 * cos_lat * (1.0 + 0.5 * seasonal) + 0.0 * LON_R, 0.0,
    )
    sw_down = np.broadcast_to(sw_down, out_shape).copy()
    lw_down = np.full(out_shape, 300.0, dtype=np.float64)
    precip = np.broadcast_to(
        3e-5 * np.exp(-(np.degrees(LAT_R) / 10.0) ** 2) + 0.0 * LON_R,
        out_shape,
    ).copy()
    runoff = np.zeros_like(precip)

    return OceanForcing(
        lon=lon, lat=lat, time_s=time_s,
        u10=u10.astype(np.float64),
        v10=v10.astype(np.float64),
        T_air=T_air.astype(np.float64),
        q_air=q_air.astype(np.float64),
        sw_down=sw_down.astype(np.float64),
        lw_down=lw_down.astype(np.float64),
        precip=precip.astype(np.float64),
        runoff=runoff.astype(np.float64),
    )


def load_jra55_do(year: int, *, cache_dir: Optional[Path] = None,
                  allow_synthetic: bool = True) -> OceanForcing:
    """Load one calendar year of JRA55-do forcing.

    Looks for ``<cache_dir>/<year>.zarr`` first; falls back to
    :func:`synthetic_ocean_forcing` when ``allow_synthetic=True`` and
    no cached year is present.
    """
    root = Path(cache_dir) if cache_dir is not None else _cache_dir()
    zarr_path = root / f"{year}.zarr"
    if zarr_path.exists():
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "JRA55-do real-data load requires xarray + zarr; install "
                "with ``pip install xarray zarr``"
            ) from exc
        ds = xr.open_zarr(zarr_path)
        return OceanForcing(
            lon=np.asarray(ds.lon.values, dtype=np.float64),
            lat=np.asarray(ds.lat.values, dtype=np.float64),
            time_s=np.asarray(ds.time_s.values, dtype=np.float64),
            u10=np.asarray(ds.u10.values, dtype=np.float64),
            v10=np.asarray(ds.v10.values, dtype=np.float64),
            T_air=np.asarray(ds.T_air.values, dtype=np.float64),
            q_air=np.asarray(ds.q_air.values, dtype=np.float64),
            sw_down=np.asarray(ds.sw_down.values, dtype=np.float64),
            lw_down=np.asarray(ds.lw_down.values, dtype=np.float64),
            precip=np.asarray(ds.precip.values, dtype=np.float64),
            runoff=np.asarray(ds.runoff.values, dtype=np.float64),
        )
    if not allow_synthetic:
        raise FileNotFoundError(
            f"JRA55-do cache missing: {zarr_path}; populate via "
            f"``scripts/download_jra55_do.py`` (see "
            f"https://climate.mri-jma.go.jp/pub/ocean/JRA55-do/)."
        )
    return synthetic_ocean_forcing(year)


__all__ = [
    "OceanForcing",
    "JRA55_NLON",
    "JRA55_NLAT",
    "JRA55_FREQ_HOURS",
    "synthetic_ocean_forcing",
    "load_jra55_do",
]
