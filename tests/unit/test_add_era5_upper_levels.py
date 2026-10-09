"""scripts/data/add_era5_upper_levels.py on synthetic local stores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "add_era5_upper_levels.py")
_spec = importlib.util.spec_from_file_location("add_era5_upper_levels", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

LON = np.arange(0.0, 360.0, 90.0)


def _field(t_hours, lev, lat):
    """Linear in latitude, so the lat-linear regrid is exact."""
    return (1000.0 * lev[None, :, None, None] + lat[None, None, :, None]
            + t_hours[:, None, None, None] + 0.0 * LON)


def test_appends_same_time_upper_levels_and_keeps_base(tmp_path):
    times = pd.date_range("1979-01-01", periods=3, freq="6h")
    src_lev = np.array([1, 10, 50, 100])
    src_lat = np.linspace(90.0, -90.0, 7)               # ERA5: north->south, poles
    th = np.arange(3) * 6.0
    src = xr.Dataset(
        {v: (("time", "level", "latitude", "longitude"),
             _field(th, src_lev, src_lat).astype("float32")) for v in mod.LEVEL_VARS},
        coords={"time": times, "level": src_lev, "latitude": src_lat, "longitude": LON})
    src.to_zarr(tmp_path / "src.zarr")

    base_lat = np.array([-80.0, -25.0, 10.0, 70.0])     # off-node, no poles
    base_lev = np.array([50, 100])
    rng = np.random.default_rng(0)
    base = xr.Dataset(
        {v: (("time", "level", "lat", "lon"),
             rng.standard_normal((1, 2, 4, 4)).astype("float32")) for v in mod.LEVEL_VARS},
        coords={"time": times[1:2], "level": base_lev, "lat": base_lat, "lon": LON})
    base["surface_pressure"] = (("time", "lat", "lon"), np.full((1, 4, 4), 1e5, "float32"))
    base.to_zarr(tmp_path / "base.zarr")

    mod.main([str(tmp_path / "base.zarr"), str(tmp_path / "out.zarr"),
              "--src", str(tmp_path / "src.zarr")])
    out = xr.open_zarr(tmp_path / "out.zarr").load()
    np.testing.assert_array_equal(out["level"], [1, 10, 50, 100])
    np.testing.assert_array_equal(out["lat"], base_lat)
    np.testing.assert_array_equal(out["surface_pressure"], base["surface_pressure"])
    for v in mod.LEVEL_VARS:
        assert out[v].dtype == np.float32 and out[v].dims == base[v].dims
        np.testing.assert_array_equal(out[v].sel(level=base_lev), base[v])
        want = _field(th[1:2], np.array([1, 10]), base_lat).astype("float32")
        np.testing.assert_allclose(out[v].sel(level=[1, 10]), want, rtol=1e-6)
