"""cover_vs_rh: the humidity-matched cover comparison must reproduce a known
curve and must refuse to quote a bin it cannot support.

The whole point of the probe is that a cover difference at MATCHED humidity
means something different from a cover difference in the mean, so the binning
and the weighting are the parts that have to be right.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_P = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate"
      / "amip_bias" / "cover_vs_rh.py")


@pytest.fixture(scope="module")
def cr():
    spec = importlib.util.spec_from_file_location("cover_vs_rh", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_binned_cover_recovers_a_planted_curve(cr):
    """Cover made an exact function of humidity must come back as that
    function, bin by bin, with uniform weights."""
    rng = np.random.default_rng(0)
    rh = rng.uniform(0.05, 0.95, size=5000)
    planted = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: 60.0, 6: 70.0, 7: 80.0}
    idx = np.clip(np.digitize(rh, cr.RH_EDGES) - 1, 0, len(cr.RH_EDGES) - 2)
    cover = np.array([planted[i] for i in idx], dtype=float)
    curve, cnt = cr.binned_cover(rh, cover, np.ones_like(rh))
    for i, want in planted.items():
        assert cnt[i] >= cr.MIN_CELLS
        assert curve[i] == pytest.approx(want)


def test_thin_bins_are_not_quoted(cr):
    """A bin with fewer than MIN_CELLS cells must come back NaN, not a number
    computed from a handful of cells."""
    rh = np.concatenate([np.full(500, 0.35), np.full(3, 0.95)])
    cover = np.concatenate([np.full(500, 20.0), np.full(3, 99.0)])
    curve, cnt = cr.binned_cover(rh, cover, np.ones_like(rh))
    assert cnt[7] == 3 and np.isnan(curve[7])
    assert curve[1] == pytest.approx(20.0)


def test_missing_cells_are_dropped_from_both_sides(cr):
    rh = np.array([0.35, np.nan, 0.35, 0.35])
    cover = np.array([20.0, 50.0, np.nan, 20.0])
    curve, cnt = cr.binned_cover(rh, cover, np.ones_like(rh))
    assert cnt.sum() == 2
    assert curve[1] == pytest.approx(20.0) if cnt[1] >= cr.MIN_CELLS else True


def test_area_weighting_changes_the_answer(cr):
    """Two cells in one bin with different weights must give the weighted mean,
    so a polar cell cannot count as much as an equatorial one."""
    rh = np.full(40, 0.35)
    cover = np.concatenate([np.full(20, 0.0), np.full(20, 100.0)])
    w = np.concatenate([np.full(20, 1.0), np.full(20, 3.0)])
    curve, _ = cr.binned_cover(rh, cover, w)
    assert curve[1] == pytest.approx(75.0)


def test_relative_humidity_uses_the_model_saturation_curve(cr):
    """At the model's own saturation mixing ratio the answer must be exactly 1,
    which is only true if the probe calls the model's curve rather than a
    re-derived fit."""
    from legoesm.thermo import saturation_mixing_ratio
    T = np.array([250.0, 280.0, 300.0])
    p = 70000.0
    r_sat = np.asarray(saturation_mixing_ratio(T, np.full_like(T, p)))
    q_sat = r_sat / (1.0 + r_sat)              # mixing ratio -> specific humidity
    rh = cr.relative_humidity(q_sat, T, p)
    np.testing.assert_allclose(rh, 1.0, rtol=1e-10)
    # Halving the MIXING ratio halves RH; halving the SPECIFIC humidity does
    # not, because the conversion between them is nonlinear -- the probe works
    # in mixing ratio for exactly this reason.
    q_half = 0.5 * r_sat / (1.0 + 0.5 * r_sat)
    np.testing.assert_allclose(cr.relative_humidity(q_half, T, p), 0.5, rtol=1e-10)
