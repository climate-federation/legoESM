import importlib.util
from pathlib import Path

import numpy as np

_D = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _D / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_layer_mean_weights_by_thickness_inside_window_only():
    C = _load("clubb_lowcloud_cesm2_compare")
    p = np.array([[60000.0, 80000.0, 90000.0, 99000.0]])
    dp = np.array([[1.0, 1.0, 3.0, 1.0]])
    x = np.array([[100.0, 1.0, 2.0, 100.0]])
    assert np.isclose(C.layer_mean(x, p, dp, 70000.0, 95000.0)[0], (1 + 6) / 4)


def test_box_mask_wraps_longitude():
    C = _load("clubb_lowcloud_cesm2_compare")
    lat = np.array([-15.0, -15.0, 5.0])
    lon = np.array([275.0, -85.0, 275.0])
    assert C.box_mask(lat, lon, (-20, -10, -90, -80)).tolist() == [True, True, False]


def test_interp_logp_any_vertical_order_and_nan_outside():
    P = _load("clubb_lowcloud_profiles")
    p = np.array([[100000.0, 85000.0, 70000.0]])
    x = np.log(p)
    out = P.interp_logp(x[:, ::-1], p[:, ::-1], np.array([85000.0, 101000.0]))
    assert np.isclose(out[0, 0], np.log(85000.0)) and np.isnan(out[0, 1])


def test_moist_top_pressure_contiguous_from_surface():
    A = _load("clubb_column_ab")
    p = np.array([[50000.0, 80000.0, 90000.0, 100000.0]] * 3)   # top-down
    rh = np.array([[0.9, 0.85, 0.9, 0.95],     # wet all the way up
                   [0.9, 0.5, 0.9, 0.95],      # dry gap at 800 hPa
                   [0.9, 0.9, 0.9, 0.5]])      # dry lowest level
    out = A.moist_top_pressure(rh, p)
    assert out[0] == 50000.0 and out[1] == 90000.0 and np.isnan(out[2])


def test_box_columns_ocean_only_and_capped(tmp_path):
    A = _load("clubb_column_ab")
    f = tmp_path / "m.npz"
    np.savez(f, lat=np.full(5, -15.0), lon=np.array([275.0, 276.0, 277.0, 278.0, 10.0]),
             land=np.array([False, True, False, False, False]))
    assert A.box_columns(f, (-20, -10, -90, -80), 10).tolist() == [0, 2, 3]
    assert len(A.box_columns(f, (-20, -10, -90, -80), 2)) == 2
