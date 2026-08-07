"""Unit tests for ``scripts/validate/ocean_fidelity/compare_three_way_nemo.py``.

``main`` does live I/O (three multi-hundred-MB snapshots plus a NEMO grid_T)
and is not exercised here, but every pure helper that decides WHICH cells get
scored is — because that is where this instrument can produce a confident wrong
number.  Each test below pins a specific finding from the adversarial review of
the first version:

* ``_coastal_mask`` padded latitude with land, so every ocean cell in the top
  and bottom rows was "coastal" even with no land anywhere near it;
* ``_nn_wet_mask`` exists because the regridder's coverage flag is a
  distance-to-wet-data flag, not a land/sea classification, and admitted land
  cells within 2.5 deg of ocean into the near-land sub-domain it was meant to
  measure;
* ``_tail`` is what separates "every column too deep" from "a few columns
  convecting to the sea floor", the distinction a bias and an RMSE cannot make.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD = (Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "ocean_fidelity" / "compare_three_way_nemo.py")


def _load():
    # The module inserts scripts/validate on sys.path at import time so it can
    # reuse the NEMO scorecard's regridder; load it by path for the same reason.
    sys.path.insert(0, str(_MOD.parents[1]))
    spec = importlib.util.spec_from_file_location("compare_three_way_nemo", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()


# --------------------------------------------------------------------------
# _coastal_mask
# --------------------------------------------------------------------------
def test_coastal_mask_all_ocean_has_no_coast():
    """REGRESSION: latitude used to be padded with land, so the polar rows of an
    entirely ocean domain were reported as near-land."""
    ocean = np.ones((10, 20), dtype=bool)
    assert m._coastal_mask(ocean, 1).sum() == 0
    assert m._coastal_mask(ocean, 2).sum() == 0


def test_coastal_mask_marks_the_ring_around_an_island():
    ocean = np.ones((11, 21), dtype=bool)
    ocean[5, 10] = False                       # one land cell, far from any edge
    coast = m._coastal_mask(ocean, 1)
    # 8-neighbour dilation of a single land cell -> the 8 surrounding cells,
    # land itself excluded (the mask is intersected with `ocean`).
    assert coast.sum() == 8
    assert coast[4, 9] and coast[6, 11]        # diagonals included
    assert not coast[5, 10]                    # the land cell itself is not ocean
    assert not coast[3, 10]                    # two cells away is not in a 1-halo


def test_coastal_mask_halo_width_is_chebyshev():
    ocean = np.ones((21, 41), dtype=bool)
    ocean[10, 20] = False
    c1 = m._coastal_mask(ocean, 1)
    c2 = m._coastal_mask(ocean, 2)
    assert c1.sum() == 3 * 3 - 1               # 3x3 block minus the land cell
    assert c2.sum() == 5 * 5 - 1               # 5x5 block: "2 cells" is a square halo
    assert c2[8, 18]                           # the far corner of the 5x5 block


def test_coastal_mask_wraps_in_longitude_only():
    ocean = np.ones((7, 9), dtype=bool)
    ocean[3, 0] = False                        # land on the western edge column
    coast = m._coastal_mask(ocean, 1)
    assert coast[3, -1]                        # x is periodic: the far column is adjacent
    # y is not periodic: land in the top row must not create coast in the bottom row
    ocean2 = np.ones((7, 9), dtype=bool)
    ocean2[0, 4] = False
    coast2 = m._coastal_mask(ocean2, 1)
    assert not coast2[-1].any()


def test_coastal_mask_rejects_zero_width():
    with pytest.raises(ValueError, match="coast-cells"):
        m._coastal_mask(np.ones((4, 4), dtype=bool), 0)


# --------------------------------------------------------------------------
# _nn_wet_mask
# --------------------------------------------------------------------------
def test_nn_wet_mask_classifies_by_nearest_source_not_by_distance_to_water():
    """A target cell whose nearest SOURCE cell is land is land, however close
    some other wet cell happens to be — the property the coverage flag lacks."""
    # Source: a 1-degree band along the equator, wet for lon < 180, dry above.
    src_lon = np.arange(0.5, 360.0, 1.0)
    src_lat = np.zeros_like(src_lon)
    src_wet = (src_lon < 180.0).astype(float)
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([90.0, 270.0, 179.6, 180.4])
    wet = m._nn_wet_mask(src_lat, src_lon, src_wet, tgt_lat, tgt_lon)[0]
    assert wet[0]                # deep inside the wet half
    assert not wet[1]            # deep inside the dry half
    assert wet[2]                # just wet-side of the boundary
    assert not wet[3]            # just dry-side, though wet water is 1 deg away


def test_nn_wet_mask_handles_the_dateline_seam():
    """Longitudes are compared on the sphere, so 359.9 and 0.1 are neighbours."""
    src_lon = np.array([0.1, 180.0])
    src_lat = np.array([0.0, 0.0])
    src_wet = np.array([1.0, 0.0])
    wet = m._nn_wet_mask(src_lat, src_lon, src_wet,
                         np.array([0.0]), np.array([359.9]))[0]
    assert wet[0]


# --------------------------------------------------------------------------
# _tail
# --------------------------------------------------------------------------
def test_tail_separates_a_broad_shift_from_a_few_runaway_columns():
    mask = np.ones(100, dtype=bool)
    nemo = np.full(100, 50.0)
    broad = nemo + 20.0                              # every column 20 m deeper
    spiky = nemo.copy(); spiky[:2] = 1050.0          # two columns hit the floor
    tb = m._tail(broad, nemo, mask, 500.0)
    ts = m._tail(spiky, nemo, mask, 500.0)
    # The two cases are constructed to have the SAME mean depth (70 m), which
    # is exactly why a bias cannot tell them apart -- and why the campaign
    # needed the median and the deep fraction to read an Arctic MLD RMSE of
    # 372 m with a bias of +88 m.
    assert np.mean(broad) == pytest.approx(np.mean(spiky))
    assert tb["median_diff"] == pytest.approx(20.0)
    assert ts["median_diff"] == pytest.approx(0.0)   # the median is untouched
    assert tb["frac_a_deeper_than"] == 0.0
    assert ts["frac_a_deeper_than"] == pytest.approx(0.02)
    assert ts["frac_b_deeper_than"] == 0.0


def test_tail_returns_none_on_an_empty_or_all_nan_domain():
    a = np.array([1.0, 2.0]); b = np.array([1.0, 2.0])
    assert m._tail(a, b, np.zeros(2, dtype=bool), 10.0) is None
    assert m._tail(np.array([np.nan, np.nan]), b,
                   np.ones(2, dtype=bool), 10.0) is None


# --------------------------------------------------------------------------
# _demean
# --------------------------------------------------------------------------
def test_demean_removes_the_area_weighted_mean_not_the_plain_mean():
    field = np.array([[0.0, 10.0]])
    area = np.array([[3.0, 1.0]])               # weight the cold cell 3x
    out = m._demean(field, area)
    assert float((area * out).sum()) == pytest.approx(0.0)
    assert out[0, 0] == pytest.approx(-2.5)     # plain-mean removal would give -5.0


def test_demean_ignores_cells_of_zero_area():
    field = np.array([[1.0, 1.0, 999.0]])
    area = np.array([[1.0, 1.0, 0.0]])
    out = m._demean(field, area)
    assert out[0, 0] == pytest.approx(0.0)


def test_demean_survives_a_nan_in_a_zero_area_cell():
    """REGRESSION: ``0.0 * nan`` is ``nan``, so weighting a NaN by zero area did
    not exclude it — one NaN made the mean NaN and voided the whole field."""
    field = np.array([[1.0, 3.0, np.nan]])
    area = np.array([[1.0, 1.0, 0.0]])
    out = m._demean(field, area)
    assert np.isfinite(out[0, :2]).all()
    assert out[0, 0] == pytest.approx(-1.0)      # mean of the two valid cells is 2
    assert np.isnan(out[0, 2])                   # the excluded cell stays NaN


def test_demean_raises_when_nothing_is_valid():
    with pytest.raises(ValueError, match="no finite cells"):
        m._demean(np.array([[np.nan, np.nan]]), np.array([[1.0, 1.0]]))
