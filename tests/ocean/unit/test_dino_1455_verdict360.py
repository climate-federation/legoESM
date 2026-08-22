"""#1455 verdict360: the 360-day two-model verdict probe's own arithmetic.

The probe owns exactly two pieces of arithmetic that are not imported from a
recorded harness -- the RSS two-sided noise floor and the 2x
INDISTINGUISHABLE rule -- plus the day grids the two sides are sampled on.
Those are what is tested here.  The probe's ``--self-check`` asserts the same
things at run time and is invoked directly so a regression fails in CI rather
than in the middle of an eight-run campaign.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def V():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import verdict360
        return verdict360
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


def test_probe_self_check_passes(V):
    assert V._self_check() == 0


def test_two_sided_floor_is_the_rss_of_the_two_measured_spreads(V):
    rows = {"lego": {i: {90: {"x": v}} for i, v in enumerate([1.0, 1.1, 0.9, 1.0])},
            "nemo": {i: {90: {"x": v}} for i, v in enumerate([2.0, 2.6, 1.4, 2.0])}}
    fl, ls, ns = V.two_sided_floor(rows, "x", 90)
    assert ls == pytest.approx(np.std([1.0, 1.1, 0.9, 1.0], ddof=1))
    assert ns == pytest.approx(np.std([2.0, 2.6, 1.4, 2.0], ddof=1))
    assert fl == pytest.approx(np.hypot(ls, ns))
    # a two-sided floor is strictly larger than either side alone -- the
    # failure mode this replaces is quoting one model's floor for a gap
    assert fl > ls and fl > ns


def test_equal_wobble_reduces_the_rss_to_the_sqrt2_difference_factor(V):
    """The 90-day lane's sqrt(2) rule is the equal-spread special case; if this
    stops holding, the two lanes are no longer using the same arithmetic."""
    a = [0.0, 1.0, 2.0, 3.0]
    rows = {"lego": {i: {90: {"x": v}} for i, v in enumerate(a)},
            "nemo": {i: {90: {"x": v + 17.0}} for i, v in enumerate(a)}}
    fl, ls, ns = V.two_sided_floor(rows, "x", 90)
    assert ls == pytest.approx(ns)
    assert fl == pytest.approx(np.sqrt(2.0) * ls)


def test_verdict_rule_accepts_inside_and_rejects_outside_both_signs(V):
    """Non-vacuity: the rule must be able to say `no`, in both directions."""
    assert V.verdict(1.99, 1.0) == "YES"
    assert V.verdict(-1.99, 1.0) == "YES"
    assert V.verdict(2.01, 1.0) == "no"
    assert V.verdict(-2.01, 1.0) == "no"
    assert V.verdict(2.0, 1.0) == "YES"          # the boundary is inclusive


def test_a_zero_floor_is_refused_not_divided_by(V):
    """An ensemble whose members never separated has a floor of exactly zero by
    construction; calling that `indistinguishable` would pass every gap."""
    assert V.verdict(1e9, 0.0) == "NO-FLOOR"
    assert V.verdict(0.0, 0.0) == "NO-FLOOR"


def test_window_mean_uses_the_pre_registered_final_90_days_on_both_sides(V):
    assert V.WINDOW_DAYS == tuple(range(280, 361, 10))
    assert len(V.WINDOW_DAYS) == 9
    rows = {"lego": {0: {d: {"x": float(d)} for d in V.WINDOW_DAYS}}}
    assert V.window_mean(rows, "lego", 0, "x") == pytest.approx(np.mean(V.WINDOW_DAYS))


def test_every_scored_day_has_a_snapshot_on_the_legoesm_side(V):
    """A scored day with no legoESM 3-D snapshot would raise deep inside the
    scoring loop, after hours of integration."""
    assert set(V.SCORE_DAYS) <= set(V.SNAP_GRID)
    assert set(V.HORIZONS) <= set(V.SCORE_DAYS)
    assert set(V.WINDOW_DAYS) <= set(V.SCORE_DAYS)


def test_scored_days_land_on_nemo_restart_dumps(V):
    """NEMO writes a restart every nn_stock=320 steps (10 days).  A scored day
    that is not a multiple of 10 has no NEMO state to compare against."""
    assert all(d % 10 == 0 for d in V.SCORE_DAYS)
    assert V.kt_of(360) == V.KT_END == 17280
    assert V.kt_of(90) == 8640


def test_the_full_section_metric_is_flagged_as_sign_mixing(V):
    """The three latitude groups carry opposite-signed gaps, so the summed
    full-section number must never be read as a per-group verdict."""
    assert "acc" in V.SIGN_MIXING
    assert set(V.SIGN_MIXING) <= set(V.KEYS)
    for k in ("g_south", "g_band", "g_north", "band", "band_c"):
        assert k in V.KEYS and k not in V.SIGN_MIXING
