"""cap_water.py: per-column log-p interpolation and the masked area mean."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias" / "cap_water.py")
_spec = importlib.util.spec_from_file_location("cap_water", _TOOL)
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)


def test_columns_to_plev_is_log_linear_and_never_extrapolates_below_the_surface():
    p_full = np.array([[1e4, 5e4, 9e4], [1e4, 5e4, 1e5]])
    f = np.log(p_full)                                   # exactly linear in log-p
    out = cw.columns_to_plev(p_full, f, np.array([3e4, 9.5e4]))
    assert out[0, 0] == pytest.approx(np.log(3e4))
    assert np.isnan(out[0, 1])                           # 950 hPa below column 0's surface (900)
    assert out[1, 1] == pytest.approx(np.log(9.5e4))


def test_run_month_uses_the_real_calendar():
    assert cw.run_month({"start_day": 0.0}, 0) == 1           # day 0 = Jan 1
    assert cw.run_month({"start_day": 0.0}, 30) == 1          # Jan 31, not "month 2"
    assert cw.run_month({"start_day": 0.0}, 31) == 2
    assert cw.run_month({"start_day": 0.0}, 45) == 2
    assert cw.run_month({"start_day": 0.0}, 59) == 3          # noleap: no Feb 29
    assert cw.run_month({"start_day": None}, 364) == 12
    assert cw.run_month({"start_day": 40.0}, 25) == 3


def test_paired_area_mean_drops_the_same_cells_on_both_sides():
    area = np.array([1.0, 1.0, 2.0]); mask = np.array([True, True, True])
    model = np.array([1.0, np.nan, 3.0]); ref = np.array([10.0, 100.0, 30.0])
    m, r, kept = cw.paired_area_mean(model, ref, area, mask)
    assert m == pytest.approx((1 + 6) / 3) and r == pytest.approx((10 + 60) / 3)   # cell 1 dropped from BOTH
    assert kept == pytest.approx(3 / 4)
    ref[2] = np.nan                                       # a missing reference value drops that cell too
    m, r, kept = cw.paired_area_mean(model, ref, area, mask)
    assert m == 1.0 and r == 10.0 and kept == pytest.approx(1 / 4)
    with pytest.raises(SystemExit):
        cw.paired_area_mean(np.full(3, np.nan), ref, area, mask)


def test_columns_to_plev_rejects_targets_above_the_top():
    p_full = np.array([[1e4, 5e4, 9e4]])
    out = cw.columns_to_plev(p_full, np.log(p_full), np.array([5e3, 2e4]))
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(np.log(2e4))


def test_area_mean_weights_and_skips_nan_and_refuses_empty():
    area = np.array([1.0, 3.0, 5.0]); mask = np.array([True, True, False])
    assert cw.area_mean(np.array([2.0, 6.0, 100.0]), area, mask) == pytest.approx((2 + 18) / 4)
    assert cw.area_mean(np.array([np.nan, 6.0, 100.0]), area, mask) == pytest.approx(6.0)
    x2 = np.array([[2.0, 1.0], [6.0, 1.0], [100.0, 1.0]])
    assert cw.area_mean(x2, area, mask).tolist() == pytest.approx([5.0, 1.0])
    with pytest.raises(SystemExit):
        cw.area_mean(np.array([np.nan, np.nan, 1.0]), area, mask)


def test_on_cells_is_periodic_in_longitude():
    import xarray as xr
    lon = np.arange(0.0, 360.0, 90.0)                     # 0, 90, 180, 270
    d = xr.DataArray(np.arange(4.0)[None, :].repeat(2, 0), dims=("lat", "lon"),
                     coords={"lat": [70.0, 80.0], "lon": lon})
    v = cw.on_cells(d, np.array([70.0, 70.0]), np.array([359.0, 271.0]))
    assert v[0] == 0.0                                      # 359 is nearest to the 0-degree column, not 270
    assert v[1] == 3.0



def test_load_state_layer_thickness_closes_the_column(tmp_path, monkeypatch):
    """dp sums to p_s - p_top, so the column-water storage term spans the whole column."""
    import json
    import numpy as np
    (tmp_path / "r").mkdir()
    vg = np.array([[200.0 / 1e5 * 1.0, 0.0, 0.0, 0.0], [0.0, 0.3, 0.7, 1.0]])   # p_top = 200 Pa * p_ref/1e5
    np.savez(tmp_path / "r" / "checkpoint_day_0015.npz", T=np.full((4, 3), 250.0),
             p_s=np.array([1e5, 9e4, 1e5, 9.5e4]), trc_q_v=np.full((4, 3), 1e-3),
             meta_vgrid=vg, day=np.array(15.0))
    json.dump({}, open(tmp_path / "r" / "experiment_config.json", "w"))
    lat = np.array([80.0, 76.0, 70.0, 60.0]); lon = np.zeros(4); area = np.ones(4)
    monkeypatch.setattr(cw.rb, "ROOT", str(tmp_path))
    monkeypatch.setattr(cw.cl, "mesh_coords", lambda exp: (lat, lon, area))
    monkeypatch.setattr(cw.cl, "cell_order", lambda z, n: np.arange(n))
    st = cw.load_state("r", 15)[0]
    from legoesm import constants
    p_top = vg[0][0] * constants.p_ref
    assert np.allclose(st["dp"].sum(axis=1), st["p_s"] - p_top)
    assert np.all(st["dp"] > 0)
