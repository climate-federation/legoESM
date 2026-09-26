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


def test_block_area_mean_is_the_cos_lat_weighted_cell_average():
    m = _load()
    lat = np.array([0.25, 0.75, 1.25, 1.75])
    x = np.repeat(np.array([1.0, 2.0, 3.0, 4.0])[:, None], 2, axis=1)
    w = np.cos(np.deg2rad(lat))
    got = m.block_area_mean(x, lat, 2)
    assert got.shape == (2, 1)
    assert abs(got[0, 0] - (1 * w[0] + 2 * w[1]) / (w[0] + w[1])) < 1e-12


def test_rain_outside_the_gpcp_record_is_refused_not_substituted():
    import pytest
    m = _load()
    import glob
    if not glob.glob(f"{m.GPCP_ROOT}/*.nc"):
        pytest.skip("GPCP file not available on this machine")
    lat = np.arange(-87.5, 90, 5.0)
    lon = np.arange(2.5, 360, 5.0)
    assert m.gpcp_pr_mm_day(lat, lon, 1979, 2) is None
    feb2002 = m.gpcp_pr_mm_day(lat, lon, 2002, 2)
    assert feb2002.shape == (36, 72) and 2.0 < np.average(
        feb2002, weights=np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 72))) < 3.2


def test_sign_agreement_counts_area_where_all_maps_share_a_sign():
    m = _load()
    a = np.array([[1.0, -1.0, 2.0, np.nan]])
    b = np.array([[3.0, 1.0, 1.0, 1.0]])
    area = np.array([[1.0, 1.0, 2.0, 5.0]])
    # finite cells: 1 (agree, w1), 2 (disagree, w1), 3 (agree, w2) -> 3/4
    assert abs(m.sign_agreement([a, b], area, np.ones_like(a, bool)) - 0.75) < 1e-12
