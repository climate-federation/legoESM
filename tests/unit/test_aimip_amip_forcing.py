"""Direct tests for the exact AIMIP-1 forcing helpers (pure temporal-interp +
lat-orientation logic). The netCDF/regrid path is a data-run concern; here we
pin the linear-interpolation and clamping contract the protocol requires.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _mod():
    spec = importlib.util.find_spec("legoesm.training.aimip_amip_forcing")
    if spec is None:
        pytest.skip("legoesm not importable")
    return importlib.import_module("legoesm.training.aimip_amip_forcing")


def test_interp_forcing_at_linear_between_months():
    m = _mod()
    # three monthly nodes at t=0,10,20 (arbitrary ns units); field ramps 0->20.
    times = np.array([0, 10, 20], dtype=np.int64)
    field = np.array([[0.0, 100.0], [10.0, 110.0], [20.0, 120.0]])
    mid = m.interp_forcing_at(times, field, 5)  # halfway 0->10
    assert mid[0] == pytest.approx(5.0)
    assert mid[1] == pytest.approx(105.0)
    q = m.interp_forcing_at(times, field, 12)  # 1/5 of the way 10->20
    assert q[0] == pytest.approx(12.0)


def test_interp_forcing_at_clamps_outside_range():
    m = _mod()
    times = np.array([0, 10, 20], dtype=np.int64)
    field = np.array([[0.0], [10.0], [20.0]])
    assert m.interp_forcing_at(times, field, -5)[0] == pytest.approx(0.0)   # clamp low
    assert m.interp_forcing_at(times, field, 999)[0] == pytest.approx(20.0)  # clamp high
    assert m.interp_forcing_at(times, field, 0)[0] == pytest.approx(0.0)     # exact node
    assert m.interp_forcing_at(times, field, 20)[0] == pytest.approx(20.0)   # exact node


def test_ghg_vmr_at_year_historical_trend():
    m = _mod()
    g79 = m.ghg_vmr_at_year(1979)
    g14 = m.ghg_vmr_at_year(2014)
    # 1979 anchor (NOAA Mauna Loa): 336.78 ppm -> VMR 336.78e-6.
    assert g79["co2"] == pytest.approx(336.78e-6, rel=1e-6)
    # transient forcing: every well-mixed GHG increases 1979 -> 2014
    for k in ("co2", "ch4", "n2o"):
        assert g14[k] > g79[k], k
    # RRTMGP VMR keys + plausible magnitudes (mole fractions)
    for k in ("co2", "ch4", "n2o", "cfc11", "cfc12"):
        assert k in g79 and 0.0 < g79[k] < 1.0e-3


def test_regrid_field_2d_flips_descending_lat():
    m = _mod()
    pytest.importorskip("scipy")

    class _Grid:  # minimal Gaussian-grid stand-in (radians)
        lat = np.deg2rad(np.array([-45.0, 0.0, 45.0]))
        lon = np.deg2rad(np.array([0.0, 180.0]))

    # ERA5-style N->S latitude with a lat-linear field; after the internal flip
    # + interp the regridded field must increase with (ascending) Gaussian lat.
    era5_lat = np.array([90.0, 0.0, -90.0])   # descending
    era5_lon = np.array([0.0, 180.0])
    field = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])  # value = f(lat), N=3 S=1
    out = m._regrid_field_2d(field, era5_lat, era5_lon, _Grid())
    col = out[:, 0]
    assert col[0] < col[1] < col[2]  # increases toward north (ascending Gaussian lat)
    assert np.all(np.isfinite(out))
