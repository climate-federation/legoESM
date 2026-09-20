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


def test_area_mean_weights_and_skips_nan_and_refuses_empty():
    area = np.array([1.0, 3.0, 5.0]); mask = np.array([True, True, False])
    assert cw.area_mean(np.array([2.0, 6.0, 100.0]), area, mask) == pytest.approx((2 + 18) / 4)
    assert cw.area_mean(np.array([np.nan, 6.0, 100.0]), area, mask) == pytest.approx(6.0)
    x2 = np.array([[2.0, 1.0], [6.0, 1.0], [100.0, 1.0]])
    assert cw.area_mean(x2, area, mask).tolist() == pytest.approx([5.0, 1.0])
    with pytest.raises(SystemExit):
        cw.area_mean(np.array([np.nan, np.nan, 1.0]), area, mask)
