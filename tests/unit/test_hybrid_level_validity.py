"""`make_hybrid_levels` must not hand the dycore NEGATIVE layer mass in silence.

`dp_k = dA_k p_ref + dB_k p_s` is linear in p_s, and the near-surface layers of
any hybrid grid have `dA_k < 0` (A must return to 0 at the ground).  So those
layers collapse and then INVERT once p_s drops far enough.  With the default
`B = eta**3` the near-surface `dB/deta -> 3`, giving

    dp > 0  <=>  p_s > (2 p_ref + p_top) / 3  ~=  667 hPa

Real orography goes well below that: a 2.5-degree AMIP run reaches p_s = 543 hPa
over the Tibetan Plateau, with 0.91% of global area under the threshold.

This is NOT diagnostics-only -- `dp_from_hybrid` feeds
`primitive_eq_latlon_cgrid`, `primitive_eq_cdgrid`, `spectral_pe` and
`primitive_eq_mpas`.
"""
import logging

import numpy as np
import pytest

from legoesm.grids.vertical import (
    hybrid_min_valid_surface_pressure,
    make_hybrid_levels,
)


def test_threshold_matches_the_analytic_value():
    """(2 p_ref + p_top)/3 for the default cubic B, approached as nlev grows."""
    analytic = (2.0 * 1.0e5 + 200.0) / 3.0
    prev = 0.0
    for nlev in (20, 40, 80, 160):
        thresh = hybrid_min_valid_surface_pressure(*_ab(nlev, 3))
        assert thresh <= analytic * 1.001
        assert thresh > prev, "must converge upward toward the eta->1 limit"
        prev = thresh
    # and the finest grid must be close, not merely bounded
    assert hybrid_min_valid_surface_pressure(*_ab(160, 3)) > 0.99 * analytic


def test_threshold_predicts_where_dp_actually_goes_negative():
    """The returned pressure must be the true sign-change point, not a guess."""
    for nlev in (20, 30, 40):
        for stretching in (0.0, 2.0):
            coord = make_hybrid_levels(nlev, stretching=stretching)
            t = hybrid_min_valid_surface_pressure(
                coord.A_half, coord.B_half, coord.p_ref)
            above = np.asarray(coord.layer_thickness_dp(np.array([t * 1.02])))
            below = np.asarray(coord.layer_thickness_dp(np.array([t * 0.98])))
            assert np.all(above > 0.0), f"nlev={nlev} st={stretching}"
            assert below.min() < 0.0, f"nlev={nlev} st={stretching}"


def test_exponent_one_threshold_is_p_top_not_zero():
    """The BUILDER's exponent=1 case, not a hand-made coordinate.

    B=eta gives A=(p_top/p_ref)(1-eta), so dA = -(p_top/p_ref) deta < 0 and
    dB = deta: the threshold is exactly p_top, tiny but NOT zero.  An earlier
    version of this test asserted 0.0 against a hand-built A=0 coordinate,
    which never exercised the builder at all (codex review 2026-07-31).
    """
    assert hybrid_min_valid_surface_pressure(*_ab(20, 1)) == pytest.approx(200.0)
    make_hybrid_levels(20, transition_exponent=1, p_s_min_Pa=1.0e4)


def test_a_coordinate_that_cannot_invert_reports_zero():
    eta = np.linspace(0.0, 1.0, 21)
    assert hybrid_min_valid_surface_pressure(np.zeros_like(eta), eta) == 0.0


def test_structurally_invalid_grids_raise_instead_of_a_false_all_clear():
    """dp <= 0 at EVERY p_s must not report a finite 'safe' pressure.

    Returning 0.0 for these would be an all-clear on a grid that is broken
    everywhere -- the exact failure mode this module guards (codex review).
    """
    eta = np.linspace(0.0, 1.0, 11)
    # dB < 0: non-monotone B
    with pytest.raises(ValueError, match="non-decreasing"):
        hybrid_min_valid_surface_pressure(np.zeros_like(eta), eta[::-1].copy())
    # dB == 0 with dA < 0: dp = dA*p_ref < 0 regardless of p_s
    with pytest.raises(ValueError, match="(?i)every surface pressure"):
        hybrid_min_valid_surface_pressure(-eta, np.zeros_like(eta))
    # dB == 0 with dA == 0: zero-thickness layers
    with pytest.raises(ValueError, match="(?i)every surface pressure"):
        hybrid_min_valid_surface_pressure(np.zeros_like(eta), np.zeros_like(eta))


def test_p_s_min_exactly_at_the_threshold_is_rejected():
    """Validity means dp > 0; at the root dp == 0, which is still degenerate."""
    from legoesm.grids.vertical import make_hybrid_levels as m
    t = hybrid_min_valid_surface_pressure(*_ab(30, 3))
    with pytest.raises(ValueError, match="invert"):
        m(30, p_s_min_Pa=t)
    m(30, p_s_min_Pa=t * 1.001)


def test_declaring_a_p_s_min_turns_it_into_a_hard_error():
    """Opt-in strictness: the caller states the range the orography needs."""
    with pytest.raises(ValueError, match="invert"):
        make_hybrid_levels(30, p_s_min_Pa=5.4e4)      # Tibet
    # and a range the default grid genuinely covers is accepted
    make_hybrid_levels(30, p_s_min_Pa=8.0e4)


def test_lower_exponent_widens_the_valid_range():
    """The documented escape hatch must actually work."""
    t3 = hybrid_min_valid_surface_pressure(
        *_ab(30, 3))
    t2 = hybrid_min_valid_surface_pressure(*_ab(30, 2))
    assert t2 < t3
    make_hybrid_levels(30, transition_exponent=2, p_s_min_Pa=5.4e4)


def _ab(nlev, exponent):
    eta = np.linspace(0.0, 1.0, nlev + 1)
    B = eta ** exponent
    return eta - B + (200.0 / 1.0e5) * (1.0 - eta), B


def test_default_config_warns_loudly(caplog):
    """Silence is the failure mode this whole module exists to prevent."""
    with caplog.at_level(logging.WARNING, logger="legoesm.grids.vertical"):
        make_hybrid_levels(30)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("NEGATIVE layer mass" in m for m in msgs), caplog.text
    # the warning must name the actual threshold, not just complain
    assert any("hPa" in m for m in msgs), caplog.text


def test_the_dycore_really_consumes_this():
    """Guard the premise: if dp_from_hybrid stopped feeding the dycore, the
    severity argument in this module's docstring would be stale."""
    import inspect

    from legoesm.atmosphere.dynamics.gcm import primitive_eq_latlon_cgrid

    src = inspect.getsource(primitive_eq_latlon_cgrid)
    assert "dp_from_hybrid(" in src, (
        "lat-lon dycore no longer calls dp_from_hybrid -- re-verify severity")
