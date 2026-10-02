"""NCAR RDA ERA5 NetCDF adapter.

NCAR's local RDA ERA5 mirror (ds633.0) is stored as one NetCDF file per
variable and period, not as a single Zarr store.  This module reads those
shared files in-place and returns the same ``ERA5Slice`` container used by the
WeatherBench/Zarr loader.
"""

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from legoesm.training.era5_to_state import ERA5Slice

_PL_CODES = {
    "z": ("128_129_z", "Z"),
    "t": ("128_130_t", "T"),
    "u": ("128_131_u", "U"),
    "v": ("128_132_v", "V"),
    "q": ("128_133_q", "Q"),
}

_SFC_CODES = {
    "sp": ("128_134_sp", "SP"),
    "skt": ("128_235_skt", "SKT"),
    "2t": ("128_167_2t", "VAR_2T"),
    "10u": ("128_165_10u", "VAR_10U"),
    "10v": ("128_166_10v", "VAR_10V"),
    "msl": ("128_151_msl", "MSL"),
}

_INV_CODES = {
    "z": ("128_129_z", "Z"),
    "lsm": ("128_172_lsm", "LSM"),
}


def _month_dir(dt: datetime) -> str:
    return dt.strftime("%Y%m")


def _day_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _hour_index(dt: datetime) -> int:
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError(f"RDA ERA5 timestamps must be exact UTC hours, got {dt!r}")
    return int(dt.hour)


def _one_match(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No RDA ERA5 file matched pattern: {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous RDA ERA5 pattern {pattern!r}: {matches[:5]}")
    return Path(matches[0])


def pressure_level_file(root: str | Path, var: str, dt: datetime) -> Path:
    """Path to one RDA pressure-level analysis file for ``var`` at ``dt``."""
    root = Path(root)
    code, _ = _PL_CODES[var]
    yyyymm = _month_dir(dt)
    yyyymmdd = _day_stamp(dt)
    pat = str(
        root
        / "e5.oper.an.pl"
        / yyyymm
        / f"e5.oper.an.pl.{code}.ll025*.{yyyymmdd}00_{yyyymmdd}23.nc"
    )
    return _one_match(pat)


def surface_file(root: str | Path, var: str, dt: datetime) -> Path:
    """Path to one RDA surface analysis file for ``var`` at ``dt``."""
    root = Path(root)
    code, _ = _SFC_CODES[var]
    yyyymm = _month_dir(dt)
    pat = str(
        root
        / "e5.oper.an.sfc"
        / yyyymm
        / f"e5.oper.an.sfc.{code}.ll025*.{yyyymm}0100_{yyyymm}*23.nc"
    )
    return _one_match(pat)


def invariant_file(root: str | Path, var: str) -> Path:
    """Path to one RDA invariant file, e.g. surface geopotential or land mask."""
    root = Path(root)
    code, _ = _INV_CODES[var]
    pat = str(
        root
        / "e5.oper.invariant"
        / "*"
        / f"e5.oper.invariant.{code}.ll025*.nc"
    )
    return _one_match(pat)


def _read_pl_var(root: str | Path, var: str, dt: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Read a pressure-level variable as ``(lat, lon, level)`` plus levels hPa."""
    path = pressure_level_file(root, var, dt)
    _, name = _PL_CODES[var]
    with xr.open_dataset(path) as ds:
        da = ds[name].isel(time=_hour_index(dt))
        data = da.transpose("latitude", "longitude", "level").values.astype(np.float32)
        levels_hpa = ds["level"].values.astype(np.float64)
    return data, levels_hpa


def _read_sfc_var(root: str | Path, var: str, dt: datetime) -> np.ndarray:
    """Read a surface variable as ``(lat, lon)``."""
    path = surface_file(root, var, dt)
    _, name = _SFC_CODES[var]
    with xr.open_dataset(path) as ds:
        return ds[name].isel(time=_hour_index(dt)).values.astype(np.float32)


def _read_invariant(root: str | Path, var: str) -> np.ndarray:
    """Read an invariant field as ``(lat, lon)``."""
    path = invariant_file(root, var)
    _, name = _INV_CODES[var]
    with xr.open_dataset(path) as ds:
        return ds[name].isel(time=0).values.astype(np.float32)


def load_rda_era5_slice(root: str | Path, dt: datetime) -> ERA5Slice:
    """Load one hourly ERA5 slice from NCAR RDA ds633.0 NetCDF files.

    Returns arrays in the same orientation as ``load_era5_slice``:
    ``(lat, lon, level)`` for pressure-level fields, descending latitude
    ``90 -> -90``, longitude ``0 -> 359.75``, pressure levels ascending in Pa.
    """
    t, levels_hpa = _read_pl_var(root, "t", dt)
    u, levels_u = _read_pl_var(root, "u", dt)
    v, levels_v = _read_pl_var(root, "v", dt)
    q, levels_q = _read_pl_var(root, "q", dt)
    if not (
        np.array_equal(levels_hpa, levels_u)
        and np.array_equal(levels_hpa, levels_v)
        and np.array_equal(levels_hpa, levels_q)
    ):
        raise ValueError("RDA pressure-level files have inconsistent level coordinates.")

    path_t = pressure_level_file(root, "t", dt)
    with xr.open_dataset(path_t) as ds:
        lat = np.deg2rad(ds["latitude"].values.astype(np.float64))
        lon = np.deg2rad(ds["longitude"].values.astype(np.float64))

    p_s = _read_sfc_var(root, "sp", dt)
    try:
        sst = _read_sfc_var(root, "skt", dt)
    except FileNotFoundError:
        sst = _read_sfc_var(root, "2t", dt)
    phis = _read_invariant(root, "z")

    return ERA5Slice(
        T=t,
        u=u,
        v=v,
        q=np.maximum(q, 0.0).astype(np.float32),
        p_s=p_s,
        sst=sst,
        phis=phis,
        lat=lat,
        lon=lon,
        plev_Pa=np.asarray(levels_hpa * 100.0, dtype=np.float64),
    )
