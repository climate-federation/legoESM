"""Prescribed land boundary (atmosphere-only AMIP): loader + periodic interpolation."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from legoesm.forcing.prescribed_land import (
    DAYS_PER_YEAR, load_prescribed_land, prescribed_land_at_day,
)

_MID = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0,
                 288.5, 319.0, 349.5])


class _Grid:
    """Minimal grid: 4 columns with lat/lon in radians (what the regridder reads)."""
    grid_lat = np.deg2rad(np.array([-60.0, -10.0, 20.0, 70.0]))
    grid_lon = np.deg2rad(np.array([10.0, 100.0, 200.0, 300.0]))


def _write(tmp_path, positive="up", times=_MID):
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    t = np.asarray(times)
    # ts: annual cycle whose phase differs per column; fluxes constant in space
    month = np.arange(t.size)[:, None, None]
    ts = 280.0 + 10.0 * np.cos(2 * np.pi * month / 12.0) + 0.0 * lat[None, :, None] + 0.0 * lon[None, None, :]
    hfss = np.full((t.size, lat.size, lon.size), 20.0)
    hfls = np.full((t.size, lat.size, lon.size), 60.0)
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), ts, {"units": "K"}),
         "hfss": (("time", "lat", "lon"), hfss, {"units": "W m-2", "positive": positive}),
         "hfls": (("time", "lat", "lon"), hfls, {"units": "W m-2", "positive": positive})},
        coords={"time": t, "lat": lat, "lon": lon})
    p = tmp_path / "presc.nc"
    ds.to_netcdf(p)
    return str(p)


def test_load_regrids_every_month_onto_the_columns(tmp_path):
    pl = load_prescribed_land(_write(tmp_path), _Grid())
    assert pl.ts.shape == (12, 4) and pl.hfss.shape == (12, 4) and pl.hfls.shape == (12, 4)
    np.testing.assert_allclose(pl.hfss, 20.0)
    np.testing.assert_allclose(pl.hfls, 60.0)
    np.testing.assert_allclose(pl.ts[0], 290.0, rtol=1e-12)


def test_interpolation_hits_anchors_and_wraps_the_year(tmp_path):
    pl = load_prescribed_land(_write(tmp_path), _Grid())
    for i, d in enumerate(_MID):
        ts, sh, lh = prescribed_land_at_day(pl, d)
        np.testing.assert_allclose(ts, pl.ts[i], rtol=1e-12)
    # between the December and January anchors across the year boundary
    ts_a, _, _ = prescribed_land_at_day(pl, 362.0)
    ts_b, _, _ = prescribed_land_at_day(pl, 362.0 + 3 * DAYS_PER_YEAR)
    np.testing.assert_allclose(ts_a, ts_b, rtol=1e-12)
    w = (362.0 - 349.5) / (15.5 + DAYS_PER_YEAR - 349.5)
    np.testing.assert_allclose(ts_a, (1 - w) * pl.ts[11] + w * pl.ts[0], rtol=1e-12)
    # strictly inside the year, on the mid-month grid
    ts_m, _, _ = prescribed_land_at_day(pl, 30.0)
    assert np.all(ts_m < pl.ts[0]) and np.all(ts_m > pl.ts[1])


def test_refuses_a_downward_flux_convention(tmp_path):
    with pytest.raises(ValueError, match="positive='up'"):
        load_prescribed_land(_write(tmp_path, positive="down"), _Grid())


def test_refuses_anchors_outside_the_year(tmp_path):
    with pytest.raises(ValueError, match="climatological year"):
        load_prescribed_land(_write(tmp_path, times=_MID + 400.0), _Grid())
