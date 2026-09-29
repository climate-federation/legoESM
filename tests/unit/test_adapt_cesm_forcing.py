"""Unit tests for ``scripts/data/adapt_cesm_forcing.py`` (CESM volcanic + ozone
-> the legoESM AMIP loaders).  Synthetic inputs, round-tripped through the REAL
loaders in ``legoesm.forcing.external``."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_PY = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "data"
       / "adapt_cesm_forcing.py")


def _mod():
    spec = importlib.util.spec_from_file_location("adapt_cesm_forcing", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _volc_ds():
    """CESM volcanic layout: bare ``month`` axis 1849-01..1851-12 (the file
    duplicates its first/last year) and a YYYYMMDD ``date``."""
    years = np.repeat([1849, 1850, 1851], 12)
    months = np.tile(np.arange(1, 13), 3)
    n = years.size
    ext = np.zeros((2, 3, 4, n))
    ext[:, :, :, :] = np.arange(n)[None, None, None, :]     # record index
    return xr.Dataset(
        {"ext_sun": (("solar_bands", "latitude", "altitude", "month"), ext),
         "ext_earth": (("terrestrial_bands", "latitude", "altitude", "month"), ext),
         "date": (("month",), years * 10000 + months * 100 + 15)},
        coords={"latitude": ("latitude", [-45.0, 0.0, 45.0]),
                "altitude": ("altitude", [10.0, 20.0, 30.0, 40.0]),
                "month": ("month", np.arange(1, n + 1),
                          {"units": "month starting from 1850 01"})})


def test_volcanic_gets_cf_time_and_the_loader_anchors_it(tmp_path):
    m = _mod()
    out = m.adapt_volcanic(_volc_ds(), 1850, 1851)
    assert out.sizes["time"] == 24
    t = out["time"].values
    assert t[0] == pytest.approx(15.5) and t[1] == pytest.approx(31 + 14.0)
    assert out["time"].attrs["calendar"] == "noleap"
    # the kept records are 1850-01.. (source index 12..), not the 1849 copy
    assert float(out["ext_sun"].isel(time=0).mean()) == 12.0
    assert "date" not in out          # stale labels must not survive next to time
    path = tmp_path / "volc.nc"
    out.to_netcdf(path)
    from legoesm.forcing.external import _load_volcanic_auto_anchored
    mid, first_date, lat, aod = _load_volcanic_auto_anchored(str(path))
    assert first_date is not None, "CF anchor must be read (not cyclic fallback)"
    assert (first_date.year, first_date.month) == (1850, 1)
    assert len(mid) == 24


def test_volcanic_rejects_gaps_and_empty_windows():
    m = _mod()
    with pytest.raises(ValueError, match="no records"):
        m.adapt_volcanic(_volc_ds(), 1900, 1901)
    gap = _volc_ds().isel(month=[0, 1, 3])
    with pytest.raises(ValueError, match="consecutive"):
        m.adapt_volcanic(gap, 1849, 1849)


def _ozone_ds():
    """CESM ozone layout on a 3-level hybrid grid, with O3 LINEAR in log(p) so
    log-p interpolation to any level inside the column is exact."""
    hyam = np.array([0.001, 0.2, 0.05])            # top .. bottom (unsorted p)
    hybm = np.array([0.0, 0.3, 0.9])
    p0 = 1.0e5
    ps = np.array([[9.0e4, 1.02e5], [1.0e5, 1.0e5]])        # (time, lat)
    p = hyam[None, None, :] * p0 + hybm[None, None, :] * ps[:, :, None]
    o3 = 1e-6 * (20.0 - np.log(p))                          # (time, lat, lev)
    return xr.Dataset(
        {"O3": (("time", "lev", "lat"), np.transpose(o3, (0, 2, 1)),
                {"units": "mol/mol"}),
         "PS": (("time", "lat"), ps), "hyam": (("lev",), hyam),
         "hybm": (("lev",), hybm), "P0": ((), p0),
         "date": (("time",), np.array([19790101, 19790106])),
         # CAM layout: time = interval END, time_bnds = the 5-day interval
         "time_bnds": (("time", "nbnd"), np.array([[-5.0, 0.0], [0.0, 5.0]]))},
        coords={"time": ("time", [0.0, 5.0],
                         {"units": "days since 1979-01-01", "calendar": "noleap"}),
                "lat": ("lat", [-10.0, 10.0]), "lev": ("lev", [1.0, 500.0, 950.0])})


def test_ozone_uses_true_pressure_and_the_loader_reads_it(tmp_path):
    m = _mod()
    out = m.adapt_ozone(_ozone_ds(), 1978, 1979)
    np.testing.assert_allclose(out["time"].values, [-2.5, 2.5])  # interval midpoints
    assert "date" not in out          # interval-END labels must not survive
    plev = out["plev"].values
    assert np.all(np.diff(plev) > 0) and out["plev"].attrs["units"] == "Pa"
    np.testing.assert_allclose(plev, np.sort([100.0, 50000.0, 95000.0]))
    # exact inside each column (linear in log p), clamped outside it
    o3 = out["O3"].values                              # (time, plev, lat)
    np.testing.assert_allclose(o3[1, :, 0], 1e-6 * (20.0 - np.log(plev)), rtol=1e-12)
    ps00 = 9.0e4                                       # column (t=0, lat=0)
    p_bottom = 0.05 * 1.0e5 + 0.9 * ps00               # 86000 Pa < 95000 target
    assert o3[0, 2, 0] == pytest.approx(1e-6 * (20.0 - np.log(p_bottom)))
    path = tmp_path / "o3.nc"
    out.to_netcdf(path)
    from legoesm.forcing.external import _load_monthly_zonal_with_levels
    mid, first_date, lat, plev_r, data = _load_monthly_zonal_with_levels(str(path), "O3")
    np.testing.assert_allclose(plev_r, plev)
    assert data.shape == (2, 2, 3)                     # (time, lat, plev)
    # the loader anchors on the MIDPOINT: 1979-01-01 minus 2.5 days
    assert (first_date.year, first_date.month, first_date.day,
            first_date.hour) == (1978, 12, 29, 12)
    np.testing.assert_allclose(mid, [0.0, 5.0])


def test_ozone_window_keeps_records_by_their_midpoint():
    m = _mod()
    # record 0 is centred on 1978-12-29 (stamped 1979-01-01 by CAM): not 1979
    out = m.adapt_ozone(_ozone_ds(), 1979, 1979)
    np.testing.assert_allclose(out["time"].values, [2.5])
    no_units = _ozone_ds()
    no_units["time"].attrs.pop("units")
    with pytest.raises(ValueError, match="units"):
        m.adapt_ozone(no_units, 1978, 1979)


def test_ozone_requires_the_hybrid_variables():
    m = _mod()
    with pytest.raises(ValueError, match="PS"):
        m.adapt_ozone(_ozone_ds().drop_vars("PS"), 1979, 1979)
