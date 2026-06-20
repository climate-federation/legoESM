"""Shared synthetic NCAR-RDA ll025 ERA5 archive writers for the offline-ERA5 tests.

The d633006 filename convention (``e5.oper.an.{pl,sfc}.<grib_code>.ll025sc.<chunk>.nc``) +
the ``T``/``U``/``V``/``Q``/``SP``/``SSTK``/``CI`` variable names and K/[0,1] units were
copy-pasted across the offline-AMIP forcing + compare tests; this is the SINGLE source of
truth for that convention (CLAUDE.md: no naming-only copy-paste).

* :func:`write_forcing_archive` — the monthly SST (``sstk``, K gradient) + sea-ice (``ci``)
  ``build_era5_amip_forcing`` reads (the FORCING side).
* :func:`write_full_archive` — that PLUS pl ``T``/``U``/``V``/``Q`` at the 13 WB2 levels +
  sfc ``SP``, so ``open_local_era5_dataset`` (the COMPARE side) and the forcing both read
  ONE archive.
"""

from __future__ import annotations

import numpy as np


def _sfc_name(code: str, day: str = "20200101") -> str:
    """The monthly single-level RDA filename for ``code`` covering ``day`` (``YYYYMMDD``)."""
    m = day[:6]
    return f"e5.oper.an.sfc.{code}.ll025sc.{m}0100_{m}3123.nc"


def _pl_name(code: str, day: str = "20200101") -> str:
    """The daily pressure-level RDA filename for ``code`` on ``day`` (``YYYYMMDD``)."""
    return f"e5.oper.an.pl.{code}.ll025sc.{day}00_{day}23.nc"


def _hourly_times(day: str, nt: int):
    start = np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:8]}T00")
    return (start + np.arange(nt) * np.timedelta64(1, "h")).astype("datetime64[ns]")


def _latlon(nlat: int, nlon: int):
    lat = np.linspace(90.0, -90.0, nlat)            # ERA5 order: descending
    lon = np.linspace(0.0, 360.0 * (nlon - 1) / nlon, nlon)
    return lat, lon


def write_forcing_archive(tmp_path, *, nlat: int = 8, nlon: int = 16,
                          nt: int = 48, sst_add: float = 0.0, day: str = "20200101") -> None:
    """Write a monthly synthetic RDA ``sstk`` + ``ci`` — the boundary fields
    ``build_era5_amip_forcing`` reads.

    ``sstk`` is a 275–300 K equator→pole SST gradient (Kelvin), optionally bumped UNIFORMLY
    by ``sst_add`` (the iter-420 consumption differential); ``ci`` is zero sea-ice ([0,1]).
    Both are monthly chunks over ``nt`` hourly times (so a daily ``hour_stride`` subsample
    clears ``build_era5_amip_forcing``'s ``>= 2`` guard).  Overwrites in place (call twice
    for a bump).
    """
    import xarray as xr

    lat, lon = _latlon(nlat, nlon)
    t = _hourly_times(day, nt)
    sst = np.broadcast_to(
        (300.0 - 25.0 * np.abs(lat) / 90.0)[:, None] + sst_add,
        (nt, nlat, nlon)).astype("f4")
    xr.Dataset({"SSTK": (("time", "latitude", "longitude"), sst, {"units": "K"})},
               coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
        tmp_path / _sfc_name("128_034_sstk", day))
    xr.Dataset({"CI": (("time", "latitude", "longitude"),
                       np.zeros((nt, nlat, nlon), "f4"), {"units": "(0-1)"})},
               coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
        tmp_path / _sfc_name("128_031_ci", day))


def write_full_archive(tmp_path, *, nlat: int = 6, nlon: int = 8,
                       nt_sfc: int = 8, day: str = "20200101") -> None:
    """Write a FULL synthetic RDA archive — pl ``T``/``U``/``V``/``Q`` at the 13 WB2
    pressure levels (constant) + sfc ``SP``/``SSTK``/``CI`` — so the offline COMPARE
    (``open_local_era5_dataset`` → ``load_era5_time_mean``) AND the FORCING
    (``build_era5_amip_forcing``) both read ONE local archive.  The pl files are daily
    chunks (the day's first 4 hourly times); the sfc files are monthly chunks.
    """
    import xarray as xr
    from legoesm.training.era5_to_state import WB2_PRESSURE_LEVELS

    lat, lon = _latlon(nlat, nlon)
    lev = np.array(WB2_PRESSURE_LEVELS, dtype="f4")     # hPa, the compare's default levels
    t_sfc = _hourly_times(day, nt_sfc)
    t_pl = t_sfc[:4]                                     # the day's hourly pl ⊂ the monthly sfc

    def _pl(code, var, val):
        data = np.full((4, len(lev), nlat, nlon), val, "f4")
        xr.Dataset(
            {var: (("time", "level", "latitude", "longitude"), data)},
            coords={"time": t_pl, "level": lev, "latitude": lat, "longitude": lon},
        ).to_netcdf(tmp_path / _pl_name(code, day))

    def _sfc(code, var, val, units=None):
        attrs = {"units": units} if units else {}
        xr.Dataset(
            {var: (("time", "latitude", "longitude"),
                   np.full((nt_sfc, nlat, nlon), val, "f4"), attrs)},
            coords={"time": t_sfc, "latitude": lat, "longitude": lon},
        ).to_netcdf(tmp_path / _sfc_name(code, day))

    _pl("128_130_t", "T", 250.0)
    _pl("128_131_u", "U", 5.0)
    _pl("128_132_v", "V", 2.0)
    _pl("128_133_q", "Q", 0.001)
    _sfc("128_134_sp", "SP", 1.0e5)
    _sfc("128_034_sstk", "SSTK", 290.0, units="K")
    _sfc("128_031_ci", "CI", 0.0, units="(0-1)")
