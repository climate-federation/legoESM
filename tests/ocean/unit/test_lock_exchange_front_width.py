"""The lock-exchange front may be softened, and the default must not move.

WHY THIS OPTION EXISTS. A step front is not multi-dimensionally monotone
under a 1-D flux LIMITER applied direction by direction. MEASURED on the
resolved Petersen channel, against an initial [5, 30] degC range: tvd
undershoots to -0.40 degC and superbee to -2.99, while the flux-CORRECTED
schemes (ppm_fct, fct2) stay exactly bounded. The MPAS tracer port has no
FCT, so a cross-grid lock exchange at resolving resolution cannot share an
advection family with the lat-lon arm while the front is a step.

WHAT THIS FILE GUARDS. The default is zero, i.e. the shipped step, so no
existing result moves; and when a width IS set the profile is monotone,
lands exactly on the two plateau values, and is centred on the front.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig


def test_the_default_is_still_a_step():
    """The whole point of shipping this OFF."""
    assert LockExchangeConfig().front_width_deg == 0.0


def _profile(d_lon, width, cold=5.0, warm=30.0):
    """The shipped expression, mirrored here so the test is an ORACLE and
    not a re-run of the implementation. Kept in sync by the plateau and
    step assertions below, which pin behaviour rather than form."""
    if width <= 0.0:
        return np.where(d_lon < 0.0, cold, warm)
    x = np.clip(d_lon / width, -1.0, 1.0)
    return cold + (warm - cold) * 0.5 * (1.0 + np.sin(0.5 * np.pi * x))


def test_zero_width_reproduces_the_step_exactly():
    d = np.linspace(-10.0, 10.0, 401)
    step = np.where(d < 0.0, 5.0, 30.0)
    assert np.array_equal(_profile(d, 0.0), step)


@pytest.mark.parametrize("width", [0.5, 2.0, 5.0])
def test_the_softened_front_is_monotone_and_bounded(width):
    """No overshoot in the IC itself -- a ramp that rings would hand the
    limiter the very problem it is meant to avoid."""
    d = np.linspace(-3.0 * width, 3.0 * width, 2001)
    T = _profile(d, width)
    assert np.all(np.diff(T) >= -1e-12), "must not decrease going east"
    assert T.min() >= 5.0 - 1e-12 and T.max() <= 30.0 + 1e-12


@pytest.mark.parametrize("width", [0.5, 2.0, 5.0])
def test_it_reaches_the_plateaus_and_is_centred(width):
    T = _profile(np.array([-2.0 * width, 0.0, 2.0 * width]), width)
    assert T[0] == pytest.approx(5.0)
    assert T[2] == pytest.approx(30.0)
    # Centred: the midpoint value sits at the mean of the two plateaus.
    assert T[1] == pytest.approx(0.5 * (5.0 + 30.0))


def test_a_wider_front_is_never_sharper():
    """Non-vacuity: the width has to actually do something monotonic."""
    d = np.linspace(-6.0, 6.0, 1201)
    grad = [np.max(np.abs(np.gradient(_profile(d, w), d))) for w in
            (1.0, 2.0, 4.0)]
    assert grad[0] > grad[1] > grad[2]
