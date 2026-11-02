"""#2303 per-column ERA5 swap helpers of heating_budget.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias/heating_budget.py"
_spec = importlib.util.spec_from_file_location("heating_budget_swap", _P)
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)

PLEV = np.array([100.0, 10000.0, 50000.0, 100000.0])


def test_swap_is_log_p_linear_and_keeps_model_where_undefined():
    prof = np.array([[200.0, 210.0, 250.0, 290.0],
                     [200.0, 210.0, np.nan, 290.0]])
    ps = np.array([101000.0, 101000.0])
    p = np.array([[np.sqrt(50000.0 * 100000.0), 100500.0, 99000.0],
                  [np.sqrt(50000.0 * 100000.0), 20000.0, 10000.0]])
    model = np.full_like(p, -1.0)
    out, keep = hb.swap_on_layers(PLEV, prof, ps, p, model)
    np.testing.assert_allclose(out[0, 0], 270.0)          # log-p midpoint
    assert keep[0, 1] and out[0, 1] == -1.0               # below 1000 hPa
    assert not keep[0, 2]
    assert keep[1, 0] and keep[1, 1]                      # NaN bracket level
    np.testing.assert_allclose(out[1, 2], 210.0)


def test_swap_keeps_model_below_era5_ground_and_refuses_above_top():
    prof = np.array([[200.0, 210.0, 250.0, 290.0]])
    out, keep = hb.swap_on_layers(PLEV, prof, np.array([60000.0]),
                                  np.array([[70000.0, 40000.0]]), np.zeros((1, 2)))
    assert keep[0, 0] and out[0, 0] == 0.0 and not keep[0, 1]
    with pytest.raises(SystemExit, match="above the top"):
        hb.swap_on_layers(PLEV, prof, np.array([1e5]), np.array([[50.0]]), np.zeros((1, 1)))


def test_snapshot_area_average_excludes_below_ground_points(tmp_path):
    xr = pytest.importorskip("xarray")
    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 180.0])
    plev = np.array([100000.0, 50000.0])                 # descending, as ERA5
    T = np.empty((1, 2, 2, 2))
    T[0, 0] = [[280.0, 282.0], [284.0, 286.0]]
    T[0, 1] = 250.0
    ps = np.array([[[101000.0, 101000.0], [101000.0, 90000.0]]])  # one point below 1000 hPa
    t = [np.datetime64("1979-01-06")]
    for name, v, fld in (("pl130", "t", T), ("pl133", "q", T * 1e-5)):
        xr.Dataset({v: (("time", "plev", "lat", "lon"), fld)},
                   coords={"time": t, "plev": plev, "lat": lat, "lon": lon}
                   ).to_netcdf(tmp_path / f"{name}_day5.nc")
    xr.Dataset({"sp": (("time", "lat", "lon"), ps)},
               coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(tmp_path / "sf134_day5.nc")
    # two cells: southern and northern hemisphere generators
    cl, cn = np.deg2rad([-45.0, 45.0]), np.deg2rad([90.0, 90.0])
    p, Tc, qc, psc = hb.era5_snapshot_on_cells(tmp_path, 5, cl, cn, "1979-01-01")
    np.testing.assert_array_equal(p, [50000.0, 100000.0])
    np.testing.assert_allclose(Tc[0], [250.0, 281.0])
    np.testing.assert_allclose(Tc[1], [250.0, 284.0])     # below-ground 286 excluded
    np.testing.assert_allclose(qc[1, 1], 284e-5)
    np.testing.assert_allclose(psc, [101000.0, 95500.0])


def test_snapshot_refuses_wrong_date(tmp_path):
    xr = pytest.importorskip("xarray")
    t = [np.datetime64("1979-01-07")]
    c = {"time": t, "lat": [0.0], "lon": [0.0]}
    for name, v in (("pl130", "t"), ("pl133", "q")):
        xr.Dataset({v: (("time", "plev", "lat", "lon"), np.ones((1, 1, 1, 1)))},
                   coords={**c, "plev": [1e5]}).to_netcdf(tmp_path / f"{name}_day5.nc")
    xr.Dataset({"sp": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
               coords=c).to_netcdf(tmp_path / "sf134_day5.nc")
    with pytest.raises(SystemExit, match="snapshot time"):
        hb.era5_snapshot_on_cells(tmp_path, 5, np.zeros(1), np.zeros(1), "1979-01-01")
