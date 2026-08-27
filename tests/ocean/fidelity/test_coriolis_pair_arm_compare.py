"""Tests for the Coriolis-pair three-arm comparison harness (#1455 action 1).

The harness adds exactly one thing to the committed scorer: an ARM AXIS.  So
the ways it can lie are (a) reducing over the wrong rows, (b) turning a
measured collapse into the wrong verdict, and (c) letting the staggering
control pass when it did not fire.  Each has a test below whose expected value
is computed independently of the code under test.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PROBE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "validate",
    "ocean_fidelity", "dino_1226", "coriolis_pair_arm_compare.py")


def _load():
    spec = importlib.util.spec_from_file_location("coriolis_pair_arm_compare",
                                                  os.path.abspath(_PROBE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


# --------------------------------------------------------------------------
# the band reduction
# --------------------------------------------------------------------------
def test_row_mask_selects_exactly_the_inclusive_band():
    wet = np.ones((10, 4), dtype=bool)
    m = M._row_mask(wet, 3, 5)
    assert m[3:6].all() and not m[:3].any() and not m[6:].any()
    assert int(m.sum()) == 3 * 4          # inclusive on BOTH ends


def test_row_mask_never_wets_a_dry_cell():
    """A band mask must be an intersection with the wet mask, not a rectangle:
    reducing over dry cells would put fabricated zeros in the RMS."""
    wet = np.ones((10, 4), dtype=bool)
    wet[4, 2] = False
    m = M._row_mask(wet, 3, 5)
    assert not m[4, 2]
    assert int(m.sum()) == 3 * 4 - 1


def test_band_reduction_isolates_a_planted_band():
    """A residual living ONLY in rows 5-6 must read zero on every other band.

    If the mask were off by a row this planted field would leak into the
    neighbouring band and the northern-lobe verdict would be reading the wrong
    latitudes.
    """
    wet = np.ones((12, 3), dtype=bool)
    series = np.zeros((5, 12, 3))
    series[:, 5:7, :] = 2.0
    inside = M.M.constancy(series, M._row_mask(wet, 5, 6))["state_mean_rms"]
    outside = M.M.constancy(series, M._row_mask(wet, 8, 10))["state_mean_rms"]
    assert inside == pytest.approx(2.0)
    assert outside == 0.0


# --------------------------------------------------------------------------
# the verdict, which must be BUILT from the values and never hardcoded
# --------------------------------------------------------------------------
# The harness's OWN rule is imported, never re-implemented.  The first version
# of this file defined a local copy of the if-chain and tested that, so
# flipping `and` to `or` in the harness left the suite green -- a test named
# for a rule that exercised only itself.
_verdict = M.verdict


def test_owner_requires_BOTH_reductions_not_either():
    """The registered rule is 'basin-wide AND the northern lobe'.

    A candidate that collapses the basin while leaving the lobe alone is the
    exact case the pair lesson says must NOT read as ownership.
    """
    assert _verdict(80.0, 80.0) == "OWNER"
    assert _verdict(80.0, 5.0) == "PARTIAL"
    assert _verdict(5.0, 80.0) == "PARTIAL"


def test_refuted_requires_both_below_the_floor():
    assert _verdict(5.0, 5.0) == "REFUTED"
    assert _verdict(5.0, 30.0) == "PARTIAL"


def test_an_overshoot_is_not_ownership():
    """Pushing the residual through zero into the opposite sign is
    over-correction; without this band it would read as a >100% OWNER."""
    assert _verdict(140.0, 140.0).startswith("OVERSHOOT")
    assert _verdict(80.0, 140.0).startswith("OVERSHOOT")


def test_the_bars_are_the_registered_ones():
    """A silently loosened bar is how a PARTIAL becomes an OWNER."""
    assert M.OWNER_BAR == 50.0
    assert M.REFUTED_BAR == 10.0
    assert M.CONTROL_BAR == 50.0


# --------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------
def test_a_control_that_gets_WORSE_reads_as_positive_percent_above_baseline():
    """A 4x worse arm is -300% collapse, which the harness must report as
    +300% ABOVE baseline.  A sign slip here would make a control that never
    fired look like it fired.  The harness's own functions are called."""
    assert M.collapse_pct(4.0, 1.0) == pytest.approx(-300.0)
    assert M.control_pct_above_baseline(4.0, 1.0) == pytest.approx(300.0)
    assert M.control_pct_above_baseline(4.0, 1.0) > M.CONTROL_BAR


def test_a_control_that_barely_moves_does_NOT_fire():
    assert M.control_pct_above_baseline(1.2, 1.0) == pytest.approx(20.0)
    assert not (M.control_pct_above_baseline(1.2, 1.0) > M.CONTROL_BAR)


def test_an_arm_that_improves_reads_as_a_POSITIVE_collapse():
    assert M.collapse_pct(0.5, 1.0) == pytest.approx(50.0)
    assert M.collapse_pct(1.0, 1.0) == pytest.approx(0.0)


def test_a_nonfinite_collapse_ABORTS_rather_than_falling_through_to_PARTIAL():
    """NaN is False against every comparison in the rule, so an empty band or
    a divide-by-nothing would read as PARTIAL -- a silent nothing wearing a
    measurement's name."""
    with pytest.raises(SystemExit, match="non-finite"):
        M.verdict(float("nan"), 80.0)
    with pytest.raises(SystemExit, match="non-finite"):
        M.verdict(80.0, float("nan"))


def test_an_empty_band_is_FATAL_not_an_empty_reduction():
    wet = np.zeros((10, 4), dtype=bool)
    wet[2:4] = True
    with pytest.raises(SystemExit, match="selects no wet cell"):
        M._row_mask(wet, 6, 8)


# --------------------------------------------------------------------------
# the arm roster
# --------------------------------------------------------------------------
def test_every_registered_arm_declares_the_stamps_it_expects():
    """The directory name is never the evidence: each arm carries the pair of
    stamps its maps must actually have, and the JOINT arm is the only one
    carrying BOTH substitutions."""
    roster = {k: (vf, cor) for k, _, vf, cor in M.ARMS}
    assert roster["BASE"] == ("none", "none")
    assert roster["E1V"] == ("nemo", "none")
    assert roster["F"] == ("none", "nemo")
    assert roster["JOINT"] == ("nemo", "nemo")
    assert roster["CTRLF"] == ("none", "stagger")


def test_the_northern_lobe_band_is_the_one_the_verdict_names():
    names = [b[0] for b in M.BANDS]
    assert any("northern lobe" in n for n in names)
    lobe = [b for b in M.BANDS if "northern lobe" in b[0]][0]
    assert (lobe[1], lobe[2]) == (185, 197)
