"""Helpers of the AMIP tropical moisture scorecard."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load():
    p = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "diag_amip_tropical_moisture.py"
    spec = importlib.util.spec_from_file_location("_amip_trop_moist", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_amip_trop_moist"] = m
    spec.loader.exec_module(m)
    return m


def test_area_mean_weights_and_skips_masked_and_nan():
    m = _load()
    x = np.array([[1.0, 3.0, np.nan]])
    w = np.array([[1.0, 3.0, 5.0]])
    assert abs(m.area_mean(x, w, np.ones_like(x, bool)) - 2.5) < 1e-12
    assert abs(m.area_mean(x, w, np.array([[True, False, True]])) - 1.0) < 1e-12


def test_budget_gap_is_zero_for_a_closed_column_and_signs_a_source():
    m = _load()
    assert m.budget_gap(3.0, 3.0, 25.0, 25.0, 30.0) == 0.0
    # rain exceeding evaporation while the column also moistens = water created
    assert abs(m.budget_gap(2.78, 2.46, 25.09, 26.18, 30.0) - (0.32 + 1.09 / 30.0)) < 1e-12


def test_band_means_select_latitude_bands():
    m = _load()
    lat = np.array([-15.0, -5.0, 5.0, 15.0])
    x = np.repeat(lat[:, None], 3, axis=1)
    area = np.ones_like(x)
    out = m.band_means(x, lat, area, np.ones_like(x, bool), [-20, -10, 0, 10, 20])
    assert out == [-15.0, -5.0, 5.0, 15.0]
