"""Shared synthetic NCAR-RDA ll025 ERA5 archive writers for the offline-ERA5 tests.

The d633006 boundary-condition filename convention
(``e5.oper.an.sfc.<grib_code>.ll025sc.<chunk>.nc``) + the ``SSTK``/``CI`` variable names and
K/[0,1] units were copy-pasted across the offline-AMIP forcing tests
(``test_offline_amip_forcing_integration`` + ``test_correction_e2e_integration``).  This
centralizes them in ONE place (CLAUDE.md: no naming-only copy-paste) so the convention has a
single source of truth.  ``build_era5_amip_forcing`` reads the monthly SST (``sstk``, K) +
sea-ice (``ci``, [0,1] fraction) chunks written here.
"""

from __future__ import annotations

import numpy as np


def write_forcing_archive(tmp_path, *, nlat: int = 8, nlon: int = 16,
                          nt: int = 48, sst_add: float = 0.0) -> None:
    """Write a monthly synthetic RDA ``sstk`` + ``ci`` — the boundary fields
    ``build_era5_amip_forcing`` reads.

    ``sstk`` is a 275–300 K equator→pole SST gradient (Kelvin), optionally bumped UNIFORMLY
    by ``sst_add`` (the iter-420 consumption differential); ``ci`` is zero sea-ice ([0,1]).
    Both are monthly chunks over ``nt`` hourly times (so a daily ``hour_stride`` subsample
    clears ``build_era5_amip_forcing``'s ``>= 2`` guard).  Overwrites in place (call twice
    for a bump).
    """
    import xarray as xr

    lat = np.linspace(90.0, -90.0, nlat)            # ERA5 order: descending
    lon = np.linspace(0.0, 360.0 * (nlon - 1) / nlon, nlon)
    t = (np.datetime64("2020-01-01T00")
         + np.arange(nt) * np.timedelta64(1, "h")).astype("datetime64[ns]")
    sst = np.broadcast_to(
        (300.0 - 25.0 * np.abs(lat) / 90.0)[:, None] + sst_add,
        (nt, nlat, nlon)).astype("f4")
    xr.Dataset({"SSTK": (("time", "latitude", "longitude"), sst, {"units": "K"})},
               coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
        tmp_path / "e5.oper.an.sfc.128_034_sstk.ll025sc.2020010100_2020013123.nc")
    xr.Dataset({"CI": (("time", "latitude", "longitude"),
                       np.zeros((nt, nlat, nlon), "f4"), {"units": "(0-1)"})},
               coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
        tmp_path / "e5.oper.an.sfc.128_031_ci.ll025sc.2020010100_2020013123.nc")
