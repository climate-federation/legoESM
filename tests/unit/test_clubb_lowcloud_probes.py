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
