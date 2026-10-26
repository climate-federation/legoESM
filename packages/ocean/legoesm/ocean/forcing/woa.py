"""World Ocean Atlas (WOA) SST climatology loader.

Reference
---------
Locarnini, R. A., et al. (2018). "World Ocean Atlas 2018, Volume 1:
Temperature", NOAA Atlas NESDIS 81.

What
----
Loads the annual-mean sea-surface temperature climatology from a
local NetCDF cache (the canonical 1 deg WOA grid is 360 x 180). When
the cache file is missing the loader raises; smoke tests opt in to a
deterministic synthetic climatology with ``allow_synthetic=True``.

Synthetic climatology
---------------------
``T_sst(lat) = 288.5 + 14 cos(lat) - 8 sin^2(lat)`` (annual mean),
plus an optional monthly seasonal-cycle modulation of 4 K. Bounded
into [263, 305] K which matches the WOA range over the global ocean.

Usage
-----
::

    from legoesm.ocean.forcing.woa import load_woa_sst
    sst_K, lat_deg, lon_deg = load_woa_sst(month=None)  # annual mean

    # Diff against the model SST for the climate-fidelity gate:
    from legoesm.ocean.diagnostics_climate import sst_climatology_bias
    res = sst_climatology_bias(sst_model_K, sst_K, area, mask)
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


def synthetic_woa_sst(*, nlon: int = WOA_LON_NATIVE,
                      nlat: int = WOA_LAT_NATIVE,
                      month: Optional[int] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic SST climatology in Kelvin."""
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    lat = np.linspace(-89.5, 89.5, nlat, dtype=np.float64)
    LAT_R = np.radians(lat)[:, None]
    cos_lat = np.cos(LAT_R)
    base = 288.5 + 14.0 * cos_lat - 8.0 * np.sin(LAT_R) ** 2
    base = np.broadcast_to(base, (nlat, nlon)).astype(np.float64)
    if month is not None:
        phase = 2.0 * np.pi * (month - 1) / 12.0
        seasonal = 4.0 * np.sin(LAT_R) * np.cos(phase)
        base = base + np.broadcast_to(seasonal, (nlat, nlon))
    return np.clip(base, 263.0, 305.0), lat, lon


def load_woa_sst(*, cache_dir: Optional[Path] = None,
                  month: Optional[int] = None,
                  allow_synthetic: bool = False,
                  nlat: int = WOA_LAT_NATIVE,
                  nlon: int = WOA_LON_NATIVE,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load WOA SST climatology in Kelvin.

    Returns
    -------
    sst_K : ndarray (n_lat, n_lon)
        Annual-mean or month-resolved SST, Kelvin.
    lat_deg : ndarray (n_lat,)
        Latitude axis, degrees North, ascending.
    lon_deg : ndarray (n_lon,)
        Longitude axis, degrees East, [0, 360).
    """
    root = Path(cache_dir) if cache_dir is not None else _cache_dir()
    nc_path = root / (
        "woa_sst_annual.nc" if month is None
        else f"woa_sst_m{month:02d}.nc"
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
        return (
            np.asarray(ds.sst.values, dtype=np.float64),
            np.asarray(ds.lat.values, dtype=np.float64),
            np.asarray(ds.lon.values, dtype=np.float64),
        )
    if not allow_synthetic:
        raise FileNotFoundError(
            f"WOA SST cache missing: {nc_path}; download from "
            "https://www.ncei.noaa.gov/products/world-ocean-atlas"
        )
    logger.warning(
        "WOA SST cache missing at %s — falling back to SYNTHETIC analytic "
        "SST. Any 'bias vs WOA' computed from it is NOT an observational "
        "comparison (allow_synthetic=True was requested).", nc_path)
    return synthetic_woa_sst(nlon=nlon, nlat=nlat, month=month)


__all__ = [
    "WOA_LON_NATIVE", "WOA_LAT_NATIVE",
    "synthetic_woa_sst", "load_woa_sst",
]
