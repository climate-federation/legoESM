"""Unit tests for legoesm.ocean.fidelity.compare."""

from __future__ import annotations

import math

import numpy as np
import pytest

from legoesm.ocean.fidelity import compare


def test_identical_fields_zero_diff():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = compare.compare_field(m, m)
    assert out.rmse == 0.0
    assert out.bias == 0.0
    assert out.nrmse == 0.0
    assert out.pattern_corr == pytest.approx(1.0)
    assert out.l2_area_weighted == 0.0
    assert out.n_valid_cells == 4


def test_constant_offset_bias_equals_offset():
    m = np.ones((3, 3)) * 5.0
    r = np.zeros((3, 3))
    out = compare.compare_field(m, r)
    # range = 0, nrmse uses _EPS denominator -> very large; check rmse + bias
    assert out.bias == pytest.approx(5.0)
    assert out.rmse == pytest.approx(5.0)
    assert out.n_valid_cells == 9


def test_nan_in_model_ignored_via_isfinite():
    m = np.array([[1.0, np.nan], [3.0, 4.0]])
    r = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = compare.compare_field(m, r)
    assert out.n_valid_cells == 3
    assert out.rmse == 0.0


def test_mask_zero_cells_excluded():
    m = np.array([[1.0, 5.0], [3.0, 4.0]])
    r = np.array([[1.0, 0.0], [3.0, 4.0]])
    mask = np.array([[1, 0], [1, 1]])
    out = compare.compare_field(m, r, mask=mask)
    assert out.n_valid_cells == 3
    assert out.rmse == 0.0


def test_weights_equal_to_ones_match_no_weights():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((10, 10))
    r = rng.standard_normal((10, 10))
    a = compare.compare_field(m, r)
    b = compare.compare_field(m, r, weights=np.ones_like(m))
    assert a.rmse == pytest.approx(b.rmse)
    assert a.bias == pytest.approx(b.bias)


def test_weights_concentrate_on_subregion():
    m = np.array([1.0, 5.0])
    r = np.array([1.0, 0.0])
    out = compare.compare_field(m, r, weights=np.array([0.0, 1.0]))
    assert out.bias == pytest.approx(5.0)
    assert out.rmse == pytest.approx(5.0)


def test_all_masked_returns_nan_and_zero_count():
    m = np.array([1.0, 2.0])
    r = np.array([1.0, 2.0])
    out = compare.compare_field(m, r, mask=np.array([0, 0]))
    assert out.n_valid_cells == 0
    assert math.isnan(out.rmse)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare.compare_field(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        compare.compare_field(np.zeros((2, 2)), np.zeros((2, 2)), mask=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        compare.compare_field(np.zeros((2, 2)), np.zeros((2, 2)), weights=np.zeros((3, 3)))


def test_pattern_correlation_anti_correlated_negative():
    m = np.array([1.0, 2.0, 3.0, 4.0])
    r = -m
    out = compare.compare_field(m, r)
    assert out.pattern_corr == pytest.approx(-1.0, rel=1e-6)


def test_l2_grows_with_more_cells_at_same_bias():
    rng = np.random.default_rng(0)
    small = rng.standard_normal((4, 4))
    large = rng.standard_normal((20, 20))
    out_s = compare.compare_field(small, small + 1.0)
    out_l = compare.compare_field(large, large + 1.0)
    assert out_l.l2_area_weighted > out_s.l2_area_weighted


# ---------------------------------------------------------------------------
# compare_zonal_mean
# ---------------------------------------------------------------------------

def test_compare_zonal_mean_identical_fields():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((20, 30))
    lat = np.linspace(-math.pi / 2 * 0.9, math.pi / 2 * 0.9, 20)
    out = compare.compare_zonal_mean(m, m, lat_rad=lat)
    assert out.rmse == 0.0
    assert out.n_valid_cells == 20


def test_compare_zonal_mean_validates_shape():
    with pytest.raises(ValueError):
        compare.compare_zonal_mean(np.zeros((3,)), np.zeros((3,)), lat_rad=np.zeros(3))
    with pytest.raises(ValueError):
        compare.compare_zonal_mean(np.zeros((3, 4)), np.zeros((3, 4)), lat_rad=np.zeros(2))


# ---------------------------------------------------------------------------
# compare_moc
# ---------------------------------------------------------------------------

def test_compare_moc_identical_fields():
    psi = np.linspace(-10, 10, 60).reshape(12, 5)
    lat = np.linspace(-1.0, 1.0, 12)
    depth = np.linspace(0.0, 4000.0, 5)
    out = compare.compare_moc(psi, psi, lat_rad=lat, depth=depth)
    assert out.rmse == 0.0


def test_compare_moc_validates_shapes():
    psi = np.zeros((4, 3))
    with pytest.raises(ValueError):
        compare.compare_moc(psi, psi, lat_rad=np.zeros(5), depth=np.zeros(3))
    with pytest.raises(ValueError):
        compare.compare_moc(psi, psi, lat_rad=np.zeros(4), depth=np.zeros(4))
    with pytest.raises(ValueError):
        compare.compare_moc(np.zeros((4,)), np.zeros((4,)), lat_rad=np.zeros(4), depth=np.zeros(1))
