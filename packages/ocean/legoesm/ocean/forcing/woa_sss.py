"""World Ocean Atlas (WOA) sea-surface-salinity climatology loader.

Reference
---------
Zweng, M. M., et al. (2018). "World Ocean Atlas 2018, Volume 2:
Salinity", NOAA Atlas NESDIS 82.

What
----
Loads the annual-mean (or monthly) WOA SSS climatology from a local
NetCDF cache on the canonical 1° grid (360 × 180).  When the cache
is missing, returns a deterministic synthetic climatology with the
broad latitudinal structure of observed SSS (subtropical maxima,
freshening towards the equator and the poles) so the matrix smoke
tests do not depend on the multi-GB WOA download.

Synthetic climatology
---------------------
``S(lat) = 34.6 + 1.4·exp(-((lat-25)/15)²) + 1.4·exp(-((lat+25)/15)²)
          − 1.5·exp(-(lat/8)²)   − 1.0·max(0, (|lat| − 60)/15)``
in PSU, clipped to ``[28.0, 37.0]``.  Approximates the
subtropical gyre maxima at ±25°, the ITCZ minimum near 0°, and
the polar freshening.

Usage
-----
::

    from legoesm.ocean.forcing.woa_sss import load_woa_sss
    sss_PSU, lat_deg, lon_deg = load_woa_sss(month=None)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


WOA_LON_NATIVE: int = 360
WOA_LAT_NATIVE: int = 180


def _cache_dir() -> Path:
    from legoesm.ocean.fidelity import cache as _cache
    return _cache.sub("obs") / "woa"


def synthetic_woa_sss(
    *,
    nlon: int = WOA_LON_NATIVE,
    nlat: int = WOA_LAT_NATIVE,
    month: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic SSS climatology [PSU].

    Mimics the leading-order zonal structure observed in WOA:
    subtropical maxima around ±25°, ITCZ minimum, polar
    freshening from runoff + sea-ice melt.

    Parameters
    ----------
    nlon, nlat : int
        Grid resolution (default WOA native 360 × 180).
    month : int or None
        1-12 for monthly resolved climatology; None for annual.
        The synthetic seasonal cycle is small (Δ ≈ 0.3 PSU) — real
        SSS is dominated by the annual mean.

    Returns
    -------
    sss_PSU : (nlat, nlon) array
    lat_deg : (nlat,) array, degrees N, ascending
    lon_deg : (nlon,) array, degrees E, [0, 360)
    """
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    lat = np.linspace(-89.5, 89.5, nlat, dtype=np.float64)
    L = lat[:, None]

    subtrop_n = 1.4 * np.exp(-((L - 25.0) / 15.0) ** 2)
    subtrop_s = 1.4 * np.exp(-((L + 25.0) / 15.0) ** 2)
    itcz_fresh = -1.5 * np.exp(-(L / 8.0) ** 2)
    polar_fresh = -1.0 * np.maximum(0.0, (np.abs(L) - 60.0) / 15.0)
    base = 34.6 + subtrop_n + subtrop_s + itcz_fresh + polar_fresh

    base = np.broadcast_to(base, (nlat, nlon)).astype(np.float64)

    if month is not None:
        phase = 2.0 * np.pi * (month - 1) / 12.0
        seasonal = 0.3 * np.sin(np.radians(L)) * np.cos(phase)
        base = base + np.broadcast_to(seasonal, (nlat, nlon))

    return np.clip(base, 28.0, 37.0), lat, lon


def load_woa_sss(
    *,
    cache_dir: Optional[Path] = None,
    month: Optional[int] = None,
    allow_synthetic: bool = False,
    nlat: int = WOA_LAT_NATIVE,
    nlon: int = WOA_LON_NATIVE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load WOA sea-surface salinity climatology [PSU].

    Looks for the cached NetCDF at
    ``<cache_dir>/woa_sss_annual.nc`` (or ``woa_sss_m<MM>.nc``).
    Raises ``FileNotFoundError`` when the cache is missing, unless
    ``allow_synthetic=True`` (smoke runs only), which returns a synthetic
    climatology with a warning.

    Returns
    -------
    sss_PSU : (n_lat, n_lon) array
    lat_deg : (n_lat,) ascending degrees north
    lon_deg : (n_lon,) degrees east in [0, 360)
    """
    root = Path(cache_dir) if cache_dir is not None else _cache_dir()
    nc_path = root / (
        "woa_sss_annual.nc" if month is None
        else f"woa_sss_m{month:02d}.nc"
    )
    if nc_path.exists():
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "WOA real-data load requires xarray; install with "
                "``pip install xarray``"
            ) from exc
        ds = xr.open_dataset(nc_path)
        # Common WOA variable names — try s_an (annual SSS) first,
        # then s_mn (monthly mean).
        var = "s_an" if "s_an" in ds.variables else "s_mn"
        sss = np.asarray(ds[var].values, dtype=np.float64)
        # WOA s_an is (time, depth, lat, lon): take the first record at
        # the surface level (index 0 on every leading axis).
        sss = sss.reshape((-1,) + sss.shape[-2:])[0]
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.mod(np.asarray(ds["lon"].values, dtype=np.float64), 360.0)
        order = np.argsort(lon)
        lon = lon[order]
        # interp_woa_sss_to_grid assumes ascending, uniform lat and a global,
        # uniform lon axis: refuse a file it would silently misread.
        if not np.all(np.diff(lat) > 0.0):
            raise ValueError(f"{nc_path}: WOA SSS latitude must be ascending")
        if not np.allclose(np.diff(lon), 360.0 / lon.size):
            raise ValueError(
                f"{nc_path}: WOA SSS longitude must be global with uniform "
                f"spacing 360/{lon.size} deg")
        return sss[:, order], lat, lon

    if not allow_synthetic:
        raise FileNotFoundError(
            f"WOA SSS cache not found at {nc_path} and "
            "allow_synthetic=False"
        )

    logger.warning(
        "WOA SSS cache missing at %s — falling back to SYNTHETIC analytic "
        "SSS. Salinity restoring and SSS scores use a made-up target. "
        "(allow_synthetic=True was requested).", nc_path)
    return synthetic_woa_sss(nlon=nlon, nlat=nlat, month=month)
