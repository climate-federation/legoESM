"""TOA albedo must be MISSING where the sun does not rise, not near-zero.

``bias_maps`` formed the albedo as rsut / max(rsdt, 1.0) on both sides.  In
polar night rsdt is ~0, so the floor returned a finite near-zero albedo instead
of a missing value, and the map drew a zonal stripe at the edge of the sunlit
region that belongs to the denominator rather than to the model.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, "scripts",
                                     "validate", "amip_bias", "bias_maps.py"))


def _load():
    spec = importlib.util.spec_from_file_location("_bias_maps_for_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_albedo_is_constant_where_lit_and_missing_where_dark():
    bm = _load()
    thr = bm.TOA_ALBEDO_MIN_RSDT
    # incoming solar sweeping through the threshold; a genuinely constant albedo
    rsdt = np.array([[0.0, 0.5, thr * 0.5, thr * 2.0, 400.0]])
    albedo_true = 0.4
    rsut = albedo_true * rsdt
    alb, valid = bm._toa_albedo(rsut, rsdt)
    assert valid.tolist() == [[False, False, False, True, True]]
    np.testing.assert_allclose(alb[valid], albedo_true, rtol=1e-12)
    assert np.all(np.isnan(alb[~valid]))
    # the specific defect: no cell may come back as a small finite albedo
    assert not np.any(np.isfinite(alb) & (alb < 0.05)), alb


def test_dark_cells_are_not_zero_which_is_what_the_floor_produced():
    """The old expression is reproduced here to show what it returned."""
    bm = _load()
    rsdt = np.array([[1e-6, 500.0]])
    rsut = np.array([[1e-7, 200.0]])
    old = rsut / np.maximum(rsdt, 1.0)          # what bias_maps used to do
    assert np.isfinite(old[0, 0]) and old[0, 0] < 1e-6   # a fake "black" cell
    new, valid = bm._toa_albedo(rsut, rsdt)
    assert np.isnan(new[0, 0]) and not valid[0, 0]
    np.testing.assert_allclose(new[0, 1], 0.4, rtol=1e-12)


def test_statistics_describe_the_same_cells_on_both_sides():
    """A one-sided gap must remove the cell from BOTH fields.

    Without this, the model panel's global mean and the reference panel's are
    averages over different regions while the bias panel uses the intersection,
    so the three printed numbers do not add up.
    """
    bm = _load()
    field = np.array([[1.0, 2.0, np.nan, 4.0]])
    ref = np.array([[1.0, np.nan, 3.0, 4.0]])
    f2, r2 = bm._common_mask(field, ref)
    keep = np.isfinite(f2)
    assert keep.tolist() == [[True, False, False, True]]
    np.testing.assert_array_equal(np.isfinite(f2), np.isfinite(r2))
    # the control: the unmasked means genuinely disagree, so the test can fail
    assert not np.isclose(np.nanmean(field), np.nanmean(ref))
    assert np.isclose(np.nanmean(f2), np.nanmean(r2))
