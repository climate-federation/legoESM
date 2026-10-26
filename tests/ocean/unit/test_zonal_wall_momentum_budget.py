"""Unit tests for the zonal-wall momentum-budget probe.

Each test fails if the behaviour it names is removed. The probe's headline is
a NULL (no alternating-sign source in the comparable terms), and a null is
exactly the kind of result a broken instrument produces for free — so the
alternation detector and the masking both have tests that prove they CAN fire.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_PROBE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "scripts", "validate", "ocean_fidelity", "dino_1226"))


def _load():
    if _PROBE_DIR not in sys.path:
        sys.path.insert(0, _PROBE_DIR)
    spec = importlib.util.spec_from_file_location(
        "zonal_wall_momentum_budget",
        os.path.join(_PROBE_DIR, "zonal_wall_momentum_budget.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("netCDF4")
zwb = _load()


def test_alternation_detector_can_fire():
    """The headline is a null, so the detector must be shown to fire."""
    assert zwb.alternation_count([1.0, -1.0, 1.0, -1.0, 1.0]) == 4
    assert zwb.alternation_count([1.0, 1.0, 1.0, 1.0, 1.0]) == 0
    assert zwb.alternation_count([1.0, -1.0, -1.0, -1.0, 1.0]) == 2


def test_alternation_ignores_exact_zeros():
    """A zero is not a sign flip; counting it as one would fake a source."""
    assert zwb.alternation_count([1.0, 0.0, 1.0]) == 0


def test_alternation_on_the_real_shape_of_a_steady_bias():
    """The measured pattern (a near-constant offset) must read as 0."""
    measured = [4.217e-9, 4.217e-9, 4.218e-9, 4.218e-9, 4.218e-9]
    assert zwb.alternation_count(measured) == 0


def _wet():
    wet = np.ones((9, 7), dtype=bool)
    wet[0, :] = False            # a land row  -> zonal wall at j=1
    wet[-1, :] = False
    wet[:, 3] = False            # a land column -> meridional wall
    return wet


def test_bands_are_disjoint_and_name_the_right_geometry():
    b = zwb.t_point_bands(_wet())
    assert (b["zonal_wall"] & b["meridional_wall"]).sum() == 0
    assert (b["zonal_wall"] & b["interior"]).sum() == 0
    assert (b["meridional_wall"] & b["interior"]).sum() == 0
    # a cell with land to the north/south only is zonal; row 1 qualifies
    assert b["zonal_wall"][1, 1]
    # a cell beside the land column is meridional, not zonal
    assert b["meridional_wall"][4, 2]
    assert not b["zonal_wall"][4, 2]


def test_volume_weights_exclude_dry_levels():
    """The defect the land control caught: a 2-D band selects COLUMNS, and a
    column wet at the surface is dry below the topography."""
    wet = _wet()
    band = np.zeros_like(wet)
    band[1, 1] = True
    area = np.ones_like(wet, dtype=float)
    e3 = np.ones(5)
    mask3 = np.ones(wet.shape + (5,), dtype=bool)
    mask3[1, 1, 3:] = False               # bottom two levels are land
    w = zwb._volume_weights(band, area, e3, mask3)
    assert w.sum() == 3.0, w
    full = zwb._volume_weights(band, area, e3,
                               np.ones(wet.shape + (5,), dtype=bool))
    assert full.sum() == 5.0


def test_weighted_rms_ignores_values_on_dry_levels():
    wet = _wet()
    band = np.zeros_like(wet)
    band[1, 1] = True
    area = np.ones_like(wet, dtype=float)
    e3 = np.ones(4)
    mask3 = np.ones(wet.shape + (4,), dtype=bool)
    mask3[1, 1, 2:] = False
    field = np.zeros(wet.shape + (4,))
    field[1, 1, :2] = 1.0
    clean = zwb.weighted_rms(field, band, area, e3, mask3)
    field[1, 1, 2:] = 1.0e6               # poison the dry levels
    poisoned = zwb.weighted_rms(field, band, area, e3, mask3)
    assert clean == poisoned == 1.0


def test_weighted_rms_refuses_a_band_with_no_wet_volume():
    wet = _wet()
    band = np.zeros_like(wet)
    band[1, 1] = True
    mask3 = np.zeros(wet.shape + (3,), dtype=bool)
    with pytest.raises(SystemExit):
        zwb.weighted_rms(np.zeros(wet.shape + (3,)), band,
                         np.ones_like(wet, dtype=float), np.ones(3), mask3)


def test_alignment_picks_the_offset_that_matches_and_reports_it():
    rng = np.random.default_rng(0)
    nemo = rng.normal(size=(6, 5, 3))
    lego = np.concatenate([rng.normal(size=(6, 1, 3)), nemo], axis=1)
    wet3 = np.ones((6, 5, 3), dtype=bool)
    out = zwb.resolve_alignment(lego, nemo, wet3, axis=1)
    assert out["offset"] == 1, out
    assert out["correlations"][1] > 0.99
    assert out["residual_ratio"] > 10.0
    assert not out["dead"]


def test_alignment_judges_the_residual_not_the_raw_gap():
    """Two offsets can both correlate above 0.98 while one is right to 4e-6.

    Subtracting the correlations hides a 4000x difference in what is left
    over, and a raw-gap guard rejected the real DINO pairing. Fails if the
    guard reverts to `best - runner_up`.
    """
    # A field SMOOTH along the shifted axis, which is what real neighbouring
    # C-grid faces look like: shifting by one still correlates ~0.98, so a
    # raw-gap guard sees a margin of ~0.02 and refuses the correct pairing.
    j = np.arange(64)[None, :, None]
    smooth = np.sin(2.0 * np.pi * j / 64.0) * np.ones((8, 1, 4))
    nemo = smooth[:, 1:61, :]
    lego = smooth[:, :61, :]                     # offset 1 is exact
    wet3 = np.ones(nemo.shape, dtype=bool)
    out = zwb.resolve_alignment(lego, nemo, wet3, axis=1)
    assert out["offset"] == 1
    assert out["correlations"][1] > 0.9999       # exact
    assert out["correlations"][0] > 0.9          # the loser still correlates
    assert out["residual_ratio"] > 10.0
    # and the RAW GAP is small -- the guard that used it refused this pairing
    assert (out["correlations"][1] - out["correlations"][0]) < 0.05


def test_alignment_refuses_a_nan_correlation():
    """A NaN left `best` pinned at its first value AND slipped past the
    refusal guard, because every comparison with NaN is False."""
    rng = np.random.default_rng(4)
    nemo = rng.normal(size=(6, 5, 3))
    lego = np.concatenate([rng.normal(size=(6, 1, 3)), nemo], axis=1)
    lego[0, 0, 0] = np.nan
    wet3 = np.ones((6, 5, 3), dtype=bool)
    with pytest.raises(SystemExit):
        zwb.resolve_alignment(lego, nemo, wet3, axis=1)


def test_alignment_refuses_when_no_offset_correlates():
    """A misaligned pairing would put every difference on the wrong cells."""
    rng = np.random.default_rng(1)
    nemo = rng.normal(size=(6, 5, 3))
    lego = rng.normal(size=(6, 6, 3))
    wet3 = np.ones((6, 5, 3), dtype=bool)
    with pytest.raises(SystemExit):
        zwb.resolve_alignment(lego, nemo, wet3, axis=1)


def test_alignment_reports_a_dead_term_instead_of_aborting():
    """A term that is identically zero has an UNDEFINED correlation. The first
    version conflated that with an invalid offset and aborted the budget."""
    nemo = np.zeros((6, 5, 3))
    lego = np.zeros((6, 6, 3))
    wet3 = np.ones((6, 5, 3), dtype=bool)
    out = zwb.resolve_alignment(lego, nemo, wet3, axis=1)
    assert out["dead"]
    assert "reason" in out


def test_verdict_inputs_marks_a_dead_term_and_does_not_divide_by_it():
    states = [{"bands": {"u:x": {
        "zonal_wall": {"dead": True, "relative_diff": None,
                       "diff_mean_ms2": 0.0},
        "interior": {"dead": True, "relative_diff": None,
                     "diff_mean_ms2": 0.0}}}}]
    out = zwb.verdict_inputs(states, [{}], {})
    assert out["u:x"]["dead_on_some_state"]


def test_per_cell_nyquist_is_not_the_band_mean_nyquist():
    """THE defect that voided the first null.

    A source whose sign varies along the wall survives a per-cell Nyquist and
    is annihilated by a band mean taken first. Fails if the aggregation order
    is swapped back.
    """
    n_states, n_cells, n_lev = 9, 40, 3
    sign_state = ((-1.0) ** np.arange(n_states))[:, None, None]
    sign_cell = ((-1.0) ** np.arange(n_cells))[None, :, None]
    amp = 1e-12
    series = amp * sign_state * sign_cell * np.ones((1, 1, n_lev))
    w = np.ones((n_cells, n_lev))
    per_cell = zwb.per_cell_nyquist(series, w)
    band_mean_first = zwb.nyquist_amplitude(
        [float(series[i].mean()) for i in range(n_states)])
    assert np.isclose(per_cell, amp, rtol=1e-9), per_cell
    assert band_mean_first < 1e-24, band_mean_first


def test_per_cell_nyquist_refuses_too_few_states_and_zero_weight():
    w = np.ones((4, 2))
    with pytest.raises(SystemExit):
        zwb.per_cell_nyquist(np.ones((2, 4, 2)), w)
    with pytest.raises(SystemExit):
        zwb.per_cell_nyquist(np.ones((5, 4, 2)), np.zeros((4, 2)))


def test_verdict_inputs_computes_the_registered_two_numbers():
    def rec(rd, dm):
        return {"dead": False, "relative_diff": rd, "diff_mean_ms2": dm}
    means = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    states = [{"bands": {"u:x": {"zonal_wall": rec(0.3, m),
                                 "interior": rec(0.1, 0.0)}}}
              for m in means]
    out = zwb.verdict_inputs(states, [{}] * len(states), {})["u:x"]
    assert np.isclose(out["enrichment_zonal_over_interior"], 3.0)
    assert out["sign_alternations"] == 8


def test_probe_prints_no_verdict_of_its_own():
    """Interpretation belongs in the analysis, not baked into the tool where
    it gets echoed back as evidence."""
    src = open(os.path.join(_PROBE_DIR,
                            "zonal_wall_momentum_budget.py")).read()
    body = src.split('"""', 2)[2]          # skip the module docstring
    for banned in ("SOURCE CONFIRMED", "=> the", "VERDICT:"):
        assert banned not in body, banned


def test_nyquist_amplitude_recovers_a_known_alternating_series():
    amp = 0.41
    n = 9
    series = [amp * (-1.0) ** i for i in range(n)]
    assert np.isclose(zwb.nyquist_amplitude(series), amp, rtol=1e-12)


def test_nyquist_amplitude_annihilates_a_steady_or_linear_series():
    assert zwb.nyquist_amplitude([3.0] * 9) < 1e-15
    assert zwb.nyquist_amplitude([1.0 + 0.5 * i for i in range(9)]) < 1e-15


def test_nyquist_sees_what_the_sign_flip_count_is_blind_to():
    """The defect this probe shipped in its first version.

    A small alternating part riding on a large steady offset never flips the
    sign, so the count reads 0 whether or not a source exists. The Nyquist
    operator recovers it. Fails if anyone reverts the source statistic to the
    sign-flip count.
    """
    steady, alt = 4.2e-9, 4.2e-11          # 1% alternating component
    series = [steady + alt * (-1.0) ** i for i in range(9)]
    assert zwb.alternation_count(series) == 0
    assert np.isclose(zwb.nyquist_amplitude(series), alt, rtol=1e-9)


def test_nyquist_amplitude_refuses_too_few_states():
    with pytest.raises(SystemExit):
        zwb.nyquist_amplitude([1.0, -1.0])


def test_verdict_inputs_reports_the_nyquist_ratio():
    def rec(rd, dm):
        return {"dead": False, "relative_diff": rd, "diff_mean_ms2": dm}
    steady, alt = 1.0, 0.01
    states = [{"bands": {"u:x": {
        "zonal_wall": rec(0.3, steady + alt * (-1.0) ** i),
        "interior": rec(0.1, 0.0)}}} for i in range(9)]
    out = zwb.verdict_inputs(states, [{}] * len(states), {})["u:x"]
    assert np.isclose(out["band_mean_nyquist_ms2"], alt, rtol=1e-6)
    assert out["sign_alternations"] == 0
    assert out["enrichment_min"] is not None
