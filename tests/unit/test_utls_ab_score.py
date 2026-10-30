"""Direct test of the A/B scoring reductions (scripts/validate/amip_bias/utls_ab_score.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias/utls_ab_score.py"
_spec = importlib.util.spec_from_file_location("utls_ab_score", _P)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_layer_pressure_hybrid():
    meta = np.array([[0.01, 0.02, 0.0], [0.0, 0.5, 1.0]])   # A/p0, B at 3 interfaces
    pf, dp = S.layer_pressure(meta, np.array([1.0e5, 6.0e4]))
    np.testing.assert_allclose(dp[0], [0.01e5 + 0.5e5, -0.02e5 + 0.5e5])
    np.testing.assert_allclose(pf[1], [0.5 * (1e3 + 2e3 + 3e4), 0.5 * (2e3 + 3e4 + 6e4)])


def test_band_layer_mean_selects_and_weights():
    lat = np.array([50.0, 60.0, 10.0])
    area = np.array([1.0, 3.0, 100.0])
    p_full = np.array([[10000.0, 30000.0]] * 3)
    dp = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]])
    f = np.array([[1.0, 99.0], [2.0, 99.0], [50.0, 99.0]])
    # only level 0 (100 hPa) of the two 45-70N cells: (1*1 + 2*6)/(1+6)
    assert S.band_layer_mean(f, p_full, dp, lat, area, (45, 70), (70, 150)) == pytest.approx(13 / 7)
    with pytest.raises(SystemExit):
        S.band_layer_mean(f, p_full, dp, lat, area, (45, 70), (500, 600))
    f[1, 0] = np.nan
    with pytest.raises(SystemExit):
        S.band_layer_mean(f, p_full, dp, lat, area, (45, 70), (70, 150))


def test_nan_outside_region_does_not_leak():
    lat = np.array([50.0, 10.0])
    area = np.array([1.0, 1.0])
    p_full = np.array([[10000.0], [10000.0]])
    dp = np.ones((2, 1))
    f = np.array([[2.0], [np.nan]])          # NaN only in the excluded tropical cell
    assert S.band_layer_mean(f, p_full, dp, lat, area, (45, 70), (70, 150)) == 2.0
