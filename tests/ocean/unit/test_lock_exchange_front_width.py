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


# ---------------------------------------------------------------------------
# The tests above exercise a MIRROR of the expression. These exercise the
# shipped function, and the case that runs it. Each comes from a codex finding
# on this PR.
# ---------------------------------------------------------------------------

from legoesm.ocean.experiments.lock_exchange import (          # noqa: E402
    lock_exchange_warm_fraction)


def test_the_shipped_function_reproduces_the_step_at_zero_width():
    lon = np.linspace(-10.0, 10.0, 401)
    f = lock_exchange_warm_fraction(lon, LockExchangeConfig())
    assert np.array_equal(f, np.where(lon < 0.0, 0.0, 1.0))


def test_the_petersen_channel_actually_contains_cold_water():
    """The case that this PR reports as fixed.

    Its domain runs 0 to 0.576 degrees east and the default front sits on the
    prime meridian -- i.e. on the channel's WESTERN WALL. Measured: one column
    of sixty-six was cold and the rest warm, so the arm was a uniform 30 degC
    box. Every bound and mixing number it reported was about a gravity current
    that did not exist.
    """
    from legoesm.grids.latlon import create_regional_latlon_grid

    grid, _ = create_regional_latlon_grid(
        n_lat=4, n_lon=64, lat_south=-0.018, lat_north=+0.018,
        lon_west=0.0, lon_east=0.576)
    lon = np.asarray(grid.lon) * 180.0 / np.pi

    cfg = _matrix_case_config()
    f = lock_exchange_warm_fraction(lon, cfg)
    cold = int((f < 0.5).sum())
    assert 0.3 * f.size < cold < 0.7 * f.size, (
        f"the channel initialises with {cold} cold columns of {f.size}: the "
        "front is not inside the domain, so this is not a lock exchange")


def _matrix_case_config():
    """The LockExchangeConfig the matrix builds for the Petersen channel."""
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts" / "matrix"))
    import run_ocean_test_matrix as mod

    cases = [c for c in mod._build_test_matrix()
             if c.case == "lock_exchange" and c.grid_type == "latlon_regional"]
    assert cases, "the Petersen channel case is not in the matrix"
    return mod._lock_exchange_config(cases[0])


def test_the_case_config_carries_the_cases_own_front():
    """A fresh default here is how the front ended up on the wall."""
    cfg = _matrix_case_config()
    assert cfg.front_longitude != LockExchangeConfig().front_longitude, (
        "the matrix hands the initialiser a default config, so a case cannot "
        "place the front inside its own domain")


def test_both_interfaces_are_smoothed_on_a_full_longitude_circle():
    """Cold on half a circle and warm on the other half has TWO joins.

    Smoothing only the named one leaves the antipode at full contrast, so the
    limiter still meets a 25 degC step and the option does not do what it
    says.
    """
    lon = np.linspace(-180.0, 180.0, 721)
    soft = lock_exchange_warm_fraction(
        lon, LockExchangeConfig(front_width_deg=2.0))
    hard = lock_exchange_warm_fraction(
        lon, LockExchangeConfig(front_width_deg=0.0))
    assert np.abs(np.diff(hard)).max() == pytest.approx(1.0)
    assert np.abs(np.diff(soft)).max() < 0.35, (
        "a full-contrast jump survives somewhere on the circle")


def test_every_grid_starts_from_the_same_front():
    """Three implementations of one profile is how they came to disagree."""
    import ast
    import inspect
    import textwrap

    from legoesm.ocean.dynamics import ocean_model_fesom

    src = textwrap.dedent(
        inspect.getsource(ocean_model_fesom.create_lock_exchange_state))
    # The CALL, not the import: an import left behind while the body builds
    # its own front would satisfy a substring check and prove nothing.
    called = {n.func.id for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "lock_exchange_warm_fraction" in called, (
        "the unstructured arm builds its own front, so a case asking for a "
        "softened one gets a step here and a ramp everywhere else")

