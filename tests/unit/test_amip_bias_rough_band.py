"""The roughness band the AMIP scorecard falls back to when the reference has holes.

ESACCI's water-path retrievals are visible/near-IR, so a January-March run has
no reference north of +45 (polar night) and none over the southern polar cap.
The scorecard answers that by taking the roughness ratio over the largest
CONTIGUOUS run of fully-covered latitude rows -- identically on model and
reference -- and printing the band.  These tests pin the three things that can
silently go wrong: the opt-in gap path, the band selection, and the promise
that a gap-free reference still produces the old global number.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_DIR = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "amip_bias")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_amipbias_{name}",
                                                  _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("regional_bias")


MLAT = np.arange(-87.5, 90.0, 5.0)
MLON = np.arange(2.5, 360.0, 5.0)


def _reference_on_own_grid(missing_north_of=None, missing_south_of=None):
    """A 1-degree reference field, NaN where the retrieval has no data."""
    rlat = np.arange(-89.5, 90.0, 1.0)
    rlon = np.arange(0.5, 360.0, 1.0)
    arr = np.ones((rlat.size, rlon.size))
    if missing_north_of is not None:
        arr[rlat > missing_north_of] = np.nan
    if missing_south_of is not None:
        arr[rlat < missing_south_of] = np.nan
    return arr, rlat, rlon


def test_gap_is_fatal_unless_the_caller_opts_in():
    arr, rlat, rlon = _reference_on_own_grid(missing_north_of=45.0)
    with pytest.raises(SystemExit):
        rb.bin_to_model(arr, rlat, rlon, MLAT, MLON, label="clwvi")


def test_opt_in_returns_nan_exactly_where_the_reference_is_missing():
    arr, rlat, rlon = _reference_on_own_grid(missing_north_of=45.0)
    out = rb.bin_to_model(arr, rlat, rlon, MLAT, MLON, label="clwvi",
                          allow_gaps=True)
    assert out.shape == (MLAT.size, MLON.size)
    empty_rows = np.isnan(out).all(axis=1)
    # every model row whose whole 5-degree span lies north of +45
    assert empty_rows.tolist() == (MLAT - 2.5 >= 45.0).tolist()
    assert np.allclose(out[~np.isnan(out)], 1.0)


def test_a_gap_free_reference_is_unchanged_by_the_opt_in():
    arr, rlat, rlon = _reference_on_own_grid()
    strict = rb.bin_to_model(arr, rlat, rlon, MLAT, MLON, label="clt")
    lax = rb.bin_to_model(arr, rlat, rlon, MLAT, MLON, label="clt",
                          allow_gaps=True)
    assert np.array_equal(strict, lax)


def test_band_is_the_measured_jfm_coverage():
    # The real JFM ESACCI clwvi hole, measured on this grid: rows 27.. empty
    # (+47.5 north), rows 0-1 empty (southern cap), 3 cells empty at row 2.
    sc = _load("scorecard")
    ref = np.ones((MLAT.size, MLON.size))
    ref[27:] = np.nan
    ref[:2] = np.nan
    ref[2, :3] = np.nan
    assert int(np.isnan(ref).sum()) == 795
    sl = sc.covered_band(ref, "clwvi")
    assert (MLAT[sl.start], MLAT[sl.stop - 1]) == (-72.5, 42.5)
    assert sl.stop - sl.start == 24


def test_a_fully_covered_reference_keeps_the_whole_grid():
    sc = _load("scorecard")
    assert sc.covered_band(np.ones((MLAT.size, MLON.size)), "clt") \
        == slice(0, MLAT.size)


def test_band_rejects_a_gap_in_the_middle():
    """min..max of the covered rows would sweep an interior hole back in."""
    sc = _load("scorecard")
    ref = np.ones((MLAT.size, MLON.size))
    ref[0:3] = np.nan          # southern cap
    ref[12, 5] = np.nan        # one interior cell -> that whole row is out
    sl = sc.covered_band(ref, "clwvi")
    assert 12 not in range(sl.start, sl.stop)
    # the longer of the two surviving runs is the one above the hole
    assert (sl.start, sl.stop) == (13, MLAT.size)


def test_a_domain_too_small_to_score_is_refused_not_quoted():
    sc = _load("scorecard")
    ref = np.full((MLAT.size, MLON.size), np.nan)
    ref[10:15] = 1.0           # 5 rows, below ROUGH_MIN_ROWS
    with pytest.raises(SystemExit):
        sc.covered_band(ref, "clwvi")


def _residual_map(f):
    """|f - 4-neighbour mean| WITHOUT any wrap: the independent oracle.

    Padded with edge repetition instead of np.roll, so the interior rows of a
    crop are computed from genuinely neighbouring data.  If the claim in
    ``covered_band``'s docstring is true, the metric's own interior agrees
    with this on every row it keeps.
    """
    p = np.pad(f, ((1, 1), (0, 0)), mode="edge")
    p = np.pad(p, ((0, 0), (1, 1)), mode="wrap")      # longitude really wraps
    n = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
    return np.abs(f - n)


def test_the_crop_changes_no_interior_residual_it_keeps():
    """The latitude wrap only touches rows the metric already discards."""
    rng = np.random.default_rng(0)
    full = rng.normal(size=(MLAT.size, MLON.size))
    sl = slice(3, 27)
    ref_full = _residual_map(full)[sl][1:-1]      # oracle, no wrap anywhere
    crop = full[sl]
    n = (np.roll(crop, 1, 0) + np.roll(crop, -1, 0)
         + np.roll(crop, 1, 1) + np.roll(crop, -1, 1)) / 4.0
    kept = np.abs(crop - n)[1:-1]                 # what the metric averages
    assert np.allclose(kept, ref_full)
    # and the wrap DOES corrupt the rows it discards, so the [1:-1] is load
    # bearing rather than decorative
    edge = np.abs(crop - n)[0]
    assert not np.allclose(edge, _residual_map(full)[sl][0])


def test_ratio_crops_both_sides_and_reports_the_band():
    sc = _load("scorecard")
    rng = np.random.default_rng(1)
    model = rng.normal(size=(MLAT.size, MLON.size))
    ref = rng.normal(size=(MLAT.size, MLON.size))
    holed = ref.copy()
    holed[27:] = np.nan
    ratio, band = sc.rough_ratio(model, holed, "clwvi")
    assert band == slice(0, 27)
    bm = _load("bias_maps")
    assert np.isclose(ratio, bm.grid_scale_residual(model[band])
                      / bm.grid_scale_residual(ref[band]))
    # a gap-free reference must take no crop and report no band
    ratio_full, band_full = sc.rough_ratio(model, ref, "clt")
    assert band_full is None
    assert np.isclose(ratio_full, bm.grid_scale_residual(model)
                      / bm.grid_scale_residual(ref))


def test_a_nan_in_the_model_field_is_fatal_not_a_dash():
    sc = _load("scorecard")
    model = np.ones((MLAT.size, MLON.size))
    model[5, 5] = np.nan
    with pytest.raises(SystemExit):
        sc.rough_ratio(model, np.ones((MLAT.size, MLON.size)), "pr")


def test_a_partially_sampled_cell_does_not_count_as_covered():
    """One valid native sample in a 5-degree box is NOT coverage."""
    arr, rlat, rlon = _reference_on_own_grid()
    # blank all but one native cell inside the model box centred at (-87.5, 2.5)
    box = (rlat < -85.0)
    inside = np.zeros_like(arr, dtype=bool)
    inside[np.ix_(box, rlon < 5.0)] = True
    arr[inside] = np.nan
    arr[np.flatnonzero(box)[0], 0] = 1.0        # leave exactly one sample
    out = rb.bin_to_model(arr, rlat, rlon, MLAT, MLON, label="clwvi",
                          allow_gaps=True)
    assert np.isnan(out[0, 0])
