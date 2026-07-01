"""Exact AIMIP-1 prescribed SST / sea-ice forcing loader.

Loads the official AIMIP Phase-1 forcing dataset
``ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc`` (Zenodo 10.5281/zenodo.17065758;
DOI record 17065758) and exposes it on the model's Gaussian grid with the
protocol's temporal convention: monthly-mean values centered at the start of
each month, **linearly interpolated** to model time (NOT the CMIP input4MIPs
mid-month algorithm, per the AIMIP spec).

Variables in the file: ``sea_surface_temperature`` [K], ``sea_ice_cover``
[fraction 0-1], ``land_sea_mask`` [land fraction]; grid 721x1440 (0.25 deg),
time 1978-10-01 .. 2025-01-01 (556 monthly steps).

Used by BOTH the AMIP inference driver and the stability fine-tune so the model
is trained and evaluated under identical, protocol-exact forcing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from legoesm.training.era5_to_state import regrid_2d_to_gaussian

# Default location the downloader (scripts) writes to; overridable.
DEFAULT_AIMIP_FORCING = (
    "results/aimip_forcing/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
)
_SST_VAR = "sea_surface_temperature"
_SIC_VAR = "sea_ice_cover"
_LSM_VAR = "land_sea_mask"


def _regrid_field_2d(field_2d, era5_lat_deg, era5_lon_deg, grid):
    """Regrid a (lat, lon) ERA5 field to the Gaussian grid (n_lat, n_lon).

    ``regrid_2d_to_gaussian`` needs ascending latitude in RADIANS; ERA5 is
    (degrees, N->S) so convert + flip. NaN (SST over land) is filled with the
    field's finite mean first so the linear interpolation stays finite at
    coastal ocean cells (true land is masked out downstream by land_sea_mask).
    """
    f = np.asarray(field_2d, dtype=np.float64)
    lat = np.asarray(era5_lat_deg, dtype=np.float64)
    lon = np.asarray(era5_lon_deg, dtype=np.float64)
    if np.any(~np.isfinite(f)):
        fill = float(np.nanmean(f))
        f = np.where(np.isfinite(f), f, fill)
    if lat[0] > lat[-1]:  # RegularGridInterpolator requires ascending axes
        lat = lat[::-1]
        f = f[::-1, :]
    return np.asarray(regrid_2d_to_gaussian(f, np.deg2rad(lat), np.deg2rad(lon), grid))


def regrid_monthly_forcing_to_gaussian(forcing_path, grid, cache_path=None):
    """Regrid every monthly SST / sea-ice / land-mask field to the Gaussian grid.

    Returns ``(times_ns, sst, sic, land)`` where ``times_ns`` is an int64 array
    of nanosecond timestamps (556,), ``sst``/``sic`` are (556, ncol), and
    ``land`` is the static land fraction (ncol,). Result is cached to
    ``cache_path`` (``.npz``) because regridding 556 0.25-deg fields is slow and
    both the run and the fine-tune reuse it.
    """
    if cache_path is not None and Path(cache_path).exists():
        d = np.load(cache_path)
        return d["times_ns"], d["sst"], d["sic"], d["land"]

    import xarray as xr
    ds = xr.open_dataset(forcing_path)
    lat_deg = ds.latitude.values
    lon_deg = ds.longitude.values
    times_ns = ds.time.values.astype("datetime64[ns]").astype(np.int64)
    n_t = ds.sizes["time"]
    ncol = len(grid.lat) * len(grid.lon)

    sst = np.empty((n_t, ncol), dtype=np.float64)
    sic = np.empty((n_t, ncol), dtype=np.float64)
    sst_da = ds[_SST_VAR].values
    sic_da = ds[_SIC_VAR].values
    for i in range(n_t):
        sst[i] = _regrid_field_2d(sst_da[i], lat_deg, lon_deg, grid).reshape(-1)
        sic[i] = np.clip(
            _regrid_field_2d(sic_da[i], lat_deg, lon_deg, grid).reshape(-1), 0.0, 1.0
        )
    land = _regrid_field_2d(ds[_LSM_VAR].values, lat_deg, lon_deg, grid).reshape(-1)
    ds.close()

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, times_ns=times_ns, sst=sst, sic=sic, land=land)
    return times_ns, sst, sic, land


def interp_forcing_at(times_ns, field, target_ns):
    """Linear-interpolate a monthly Gaussian field (n_t, ncol) to ``target_ns``.

    ``target_ns`` is a scalar int64 ns timestamp. Values outside the monthly
    range clamp to the endpoints (np.interp semantics). Returns (ncol,).
    """
    t = np.asarray(times_ns, dtype=np.float64)
    tt = float(target_ns)
    if tt <= t[0]:
        return np.asarray(field[0], dtype=np.float64)
    if tt >= t[-1]:
        return np.asarray(field[-1], dtype=np.float64)
    j = int(np.searchsorted(t, tt))  # t[j-1] < tt <= t[j]
    w = (tt - t[j - 1]) / (t[j] - t[j - 1])
    return (1.0 - w) * np.asarray(field[j - 1]) + w * np.asarray(field[j])
