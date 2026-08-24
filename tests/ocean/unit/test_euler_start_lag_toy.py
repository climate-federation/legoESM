"""Known-answer tests for the Euler-start half-step toy.

The toy is what turns "an Euler first step delays the wave" from an assertion
into a checkable claim, so its own claims get controls.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PROBE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "scripts", "validate", "ocean_fidelity", "dino_1226",
    "euler_start_lag_toy.py")


def _load():
    spec = importlib.util.spec_from_file_location("euler_start_lag_toy", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


toy = _load()


@pytest.mark.parametrize("period", [6, 8, 12, 20, 40, 80])
def test_euler_start_is_exactly_the_two_point_running_mean(period):
    """THE IDENTITY. Unfiltered, euler[k] == (lf[k-1] + lf[k]) / 2 exactly.

    This is the whole mechanism: a two-point running mean is a half-sample
    delay at every frequency, which is why the measured lag was broadband and
    frequency-flat instead of looking like a wave-speed error.
    """
    assert toy.running_mean_residual(period) < 1e-12


def test_unfiltered_lag_is_exactly_half_a_step_at_every_period():
    stable = [p for p in toy.TOY_PERIODS if p >= toy.STABLE_MIN_PERIOD]
    res, _ = toy.fit_alpha(stable, 0.0)
    assert res["alpha_steps"] == pytest.approx(0.5, abs=1e-9)
    assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)
    for period in stable:
        one, _ = toy.fit_alpha([period], 0.0)
        assert one["alpha_steps"] == pytest.approx(0.5, abs=1e-9)


def test_identical_starts_give_exactly_zero_lag():
    """CONTROL. Without the Euler start there is no lag to find.

    This is also why the twin's free lane could not catch the defect: with no
    wave launched, there is no phase for a launch error to shift.
    """
    stable = [p for p in toy.TOY_PERIODS if p >= toy.STABLE_MIN_PERIOD]
    _, ewt = toy.fit_alpha(stable, 0.1)
    same = np.stack([toy.run_mode(p, 0.1, False) for p in stable], 1)[:, None, :]
    res = ewt.substep_lag_fit(same, same, np.ones((1, len(stable)), dtype=bool))
    assert res["alpha_steps"] == pytest.approx(0.0, abs=1e-12)


def test_asselin_tilts_the_lag_up_and_makes_it_period_dependent():
    """gamma=0.1 is what the card runs, and it is NOT the flat 0.5.

    NEMO filters from step 1 while legoESM's Euler branch does not
    (ocean_model_latlon_cgrid.py:8058), and that asymmetry raises alpha and
    makes it rise with period -- which is what the twin's per-band fit saw.
    """
    got = {p: toy.fit_alpha([p], 0.1)[0]["alpha_steps"]
           for p in (6, 8, 12, 20, 40, 80)}
    assert got[6] == pytest.approx(0.504, abs=0.002)
    assert got[80] == pytest.approx(0.555, abs=0.002)
    # strictly increasing with period, and everywhere above the unfiltered 0.5
    vals = [got[p] for p in (6, 8, 12, 20, 40, 80)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert all(v > 0.5 for v in vals)


def test_short_periods_are_unstable_and_are_excluded():
    """SYNTHETIC VIOLATION for the toy's own scope.

    A broadband number that includes the unstable columns is meaningless: the
    filtered leap-frog reaches ~1e25 at a 4-step period over 159 samples. The
    toy quotes periods >= 6 for exactly this reason, and this test fails if
    someone widens that range.
    """
    assert np.abs(toy.run_mode(4, 0.1, False)).max() > 1e20
    assert np.abs(toy.run_mode(5, 0.1, False)).max() > 1e10
    assert np.abs(toy.run_mode(toy.STABLE_MIN_PERIOD, 0.1, False)).max() < 10.0


def test_run_mode_refuses_a_period_at_or_below_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        toy.run_mode(2.0, 0.1, False)


def test_main_runs():
    assert toy.main([]) == 0
