from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_p = Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_monthly_climatology.py"
_spec = importlib.util.spec_from_file_location("build_monthly_climatology", _p)
bmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bmc)


def _ds(units, n_years=3, start=0.0):
    t = start + np.arange(12 * n_years) + (0.5 if units.startswith("months") else 0.0)
    if units.startswith("days"):
        t = 15.0 + 30.4375 * np.arange(12 * n_years) + start
    rng = np.random.default_rng(0)
    shp = (t.size, 2, 3)
    return xr.Dataset(
        {"aod": (("time", "band", "lat"), rng.uniform(0.01, 1.0, shp)),
         "ssa": (("time", "band", "lat"), rng.uniform(0.5, 1.0, shp)),
         "asy": (("time", "band", "lat"), rng.uniform(0.3, 0.9, shp)),
         "lat": (("lat",), np.array([-10.0, 0.0, 10.0]))},
        coords={"time": ("time", t, {"units": units, "calendar": "noleap"})})


@pytest.mark.parametrize("units", ["months since 2004-01-01", "days since 2004-01-01"])
def test_plain_monthly_mean_over_the_span(units):
    ds = _ds(units)                                     # 2004, 2005, 2006
    out = bmc.climatology(ds, 2005, 2006)
    assert out.sizes["time"] == 12
    np.testing.assert_array_equal(out["time"].values, ds["time"].values[12:24])
    want = 0.5 * (ds["asy"].values[12:24] + ds["asy"].values[24:36])
    np.testing.assert_allclose(out["asy"].values, want, rtol=1e-12)
    np.testing.assert_array_equal(out["lat"].values, ds["lat"].values)


def test_optics_weighting():
    ds = _ds("days since 2004-01-01")
    out = bmc.climatology(ds, 2005, 2006, optics=True)
    a, s, g = (ds[v].values[[12, 24], 0, 0] for v in ("aod", "ssa", "asy"))
    np.testing.assert_allclose(out["aod"].values[0, 0, 0], a.mean(), rtol=1e-12)
    np.testing.assert_allclose(out["ssa"].values[0, 0, 0], (a * s).sum() / a.sum(), rtol=1e-12)
    np.testing.assert_allclose(out["asy"].values[0, 0, 0],
                               (a * s * g).sum() / (a * s).sum(), rtol=1e-12)


def test_refuses_a_span_off_the_file():
    with pytest.raises(ValueError, match="expected 2"):
        bmc.climatology(_ds("days since 2004-01-01"), 2006, 2007)


def test_time_bounds_follow_the_first_year_stamps():
    ds = _ds("days since 2004-01-01")
    ds["time"].attrs["bounds"] = "time_bnds"
    t = ds["time"].values
    ds["time_bnds"] = (("time", "nb"), np.stack([t - 15.0, t + 15.0], -1))
    out = bmc.climatology(ds, 2005, 2006)
    np.testing.assert_array_equal(out["time_bnds"].values, ds["time_bnds"].values[12:24])
