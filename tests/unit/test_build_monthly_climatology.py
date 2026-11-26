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
    with pytest.raises(ValueError, match="2007-01: 0 records"):
        bmc.climatology(_ds("days since 2004-01-01"), 2006, 2007)


def test_a_duplicate_month_cannot_hide_a_missing_one():
    """2005-03 doubled, 2006-03 missing: each month still has 2 records in
    2005-2006, so only a per-(year, month) check sees it."""
    ds = _ds("months since 2004-01-01")
    t = ds["time"].values.copy()
    t[12 + 12 + 2] = t[12 + 2]          # 2006-03 restamped as 2005-03
    ds = ds.assign_coords(time=("time", t, ds["time"].attrs))
    with pytest.raises(ValueError, match="2005-03: 2 records"):
        bmc.climatology(ds, 2005, 2006)


def test_time_bounds_follow_the_first_year_stamps():
    ds = _ds("days since 2004-01-01")
    ds["time"].attrs["bounds"] = "time_bnds"
    t = ds["time"].values
    ds["time_bnds"] = (("time", "nb"), np.stack([t - 15.0, t + 15.0], -1))
    out = bmc.climatology(ds, 2005, 2006)
    np.testing.assert_array_equal(out["time_bnds"].values, ds["time_bnds"].values[12:24])


def test_stamps_are_the_first_years_whatever_the_file_order():
    ds = _ds("days since 2004-01-01")
    want = ds["time"].values[12:24]
    out = bmc.climatology(ds.isel(time=np.random.default_rng(1).permutation(36)), 2005, 2006)
    np.testing.assert_array_equal(np.sort(out["time"].values), want)


def test_optics_weights_skip_missing_values():
    ds = _ds("days since 2004-01-01")
    ds["ssa"].values[12, 0, 0] = np.nan                  # 2005-01 missing at one point
    out = bmc.climatology(ds, 2005, 2006, optics=True)
    np.testing.assert_allclose(out["ssa"].values[0, 0, 0], ds["ssa"].values[24, 0, 0],
                               rtol=1e-12)               # the 2006 value alone
    ds["ssa"].values[24, 0, 0] = np.nan                  # both years missing
    assert np.isnan(bmc.climatology(ds, 2005, 2006, optics=True)["ssa"].values[0, 0, 0])


def test_months_since_needs_a_day_one_reference():
    with pytest.raises(ValueError, match="day 1"):
        bmc.climatology(_ds("months since 2004-01-16"), 2005, 2006)


def test_an_integer_field_keeps_its_fractional_mean():
    ds = _ds("days since 2004-01-01")
    ds["n"] = ("time", np.arange(36) % 2)      # 2005-01 = 0, 2006-01 = 0 ... alternating
    ds["n"].values[12], ds["n"].values[24] = 1, 2
    assert bmc.climatology(ds, 2005, 2006)["n"].values[0] == 1.5


def test_months_since_needs_a_midnight_reference():
    with pytest.raises(ValueError, match="00:00 on day 1"):
        bmc.climatology(_ds("months since 2004-01-01 12:00:00"), 2005, 2006)


def test_bounds_name_the_month_a_record_covers():
    """Stamps at each month's END (the next month's day 1) with bounds
    spanning the month: grouping by stamp would average December into
    January."""
    ds = _ds("months since 2004-01-01")
    lo = np.arange(36)                                   # an INTEGER time axis
    ds = ds.assign_coords(time=("time", lo + 1, ds["time"].attrs))
    ds["time"].attrs["bounds"] = "time_bnds"
    ds["time_bnds"] = (("time", "nb"), np.stack([lo, lo + 1], -1))
    out = bmc.climatology(ds, 2005, 2006)
    want = 0.5 * (ds["aod"].values[12] + ds["aod"].values[24])     # Jan 2005, Jan 2006
    np.testing.assert_allclose(out["aod"].values[0], want, rtol=1e-12)
    assert out["time"].values[0] == 12.5                            # mid-January 2005
    del ds["time_bnds"]                                             # named but absent
    with pytest.raises(ValueError, match="does not carry"):
        bmc.climatology(ds, 2005, 2006)
