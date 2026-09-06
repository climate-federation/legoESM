"""Decision 17: the mask guard sits BELOW the linear-free-surface return.

The user answered this one "Yes": a linear-free-surface caller returns
``e3t_0`` untouched and never reads a mask, so refusing it for a missing mask
was a defect, not a check.  The moving-thickness path is unchanged and still
refuses a missing mask -- that is the whole point of the guard, and the third
arm below proves it still fires.

The second arm is the one that matters for regression: it reads the source and
goes red if the mask guard is moved back ABOVE the early return.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.vertical import nemo_qco_live_t_thickness


class _ZCoord:
    """The smallest thing the function reads: three optional attributes."""

    def __init__(self, *, nemo_e3t_0=None, is_active=None,
                 linear_free_surface=False):
        self.nemo_e3t_0 = nemo_e3t_0
        self.is_active = is_active
        self.linear_free_surface = linear_free_surface


def _e3t_0(nz=3):
    return jnp.asarray(np.arange(1.0, nz + 1.0)[None, None, :])


def test_a_linear_free_surface_caller_with_no_mask_is_served(dtype=jnp.float64):
    """DECISION 17.  This raised before; it returns e3t_0 now."""
    z = _ZCoord(nemo_e3t_0=_e3t_0(), is_active=None, linear_free_surface=True)
    out = nemo_qco_live_t_thickness(
        jnp.zeros((1, 1)), jnp.ones((1, 1)), z, jnp.float64)
    assert np.array_equal(np.asarray(out), np.asarray(_e3t_0()))


def test_a_moving_thickness_caller_with_no_mask_is_STILL_refused():
    """The half of the guard that did not move must still fire."""
    z = _ZCoord(nemo_e3t_0=_e3t_0(), is_active=None, linear_free_surface=False)
    with pytest.raises(ValueError, match="is_active"):
        nemo_qco_live_t_thickness(
            jnp.zeros((1, 1)), jnp.ones((1, 1)), z, jnp.float64)


def test_a_caller_with_no_reference_thickness_is_refused_on_both_paths():
    """The e3t_0 half stays ABOVE the early return: the return needs it."""
    for linear in (True, False):
        z = _ZCoord(nemo_e3t_0=None, is_active=jnp.ones((1, 1, 3)),
                    linear_free_surface=linear)
        with pytest.raises(ValueError, match="nemo_e3t_0"):
            nemo_qco_live_t_thickness(
                jnp.zeros((1, 1)), jnp.ones((1, 1)), z, jnp.float64)


def test_the_mask_guard_is_below_the_early_return_in_the_source():
    """Non-vacuity: moving the guard back above the return turns this red.

    A behavioural test cannot see the ORDER, because a linear-free-surface
    caller that supplies a mask is served either way.  This reads the source
    and pins the order directly.
    """
    lines = inspect.getsource(nemo_qco_live_t_thickness).splitlines()
    early_return = next(
        i for i, line in enumerate(lines)
        if "linear_free_surface" in line and "getattr" in line)
    mask_guard = next(
        i for i, line in enumerate(lines)
        if line.strip() == "if active is None:")
    tmask = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("tmask = jnp.asarray(active"))
    assert mask_guard > early_return, (
        "the is_active guard moved back above the linear-free-surface early "
        "return; decision 17 put it below")
    assert tmask > early_return, (
        "the mask conversion moved back above the early return, which would "
        "raise a TypeError before the guard could give its own message")
    # A duplicate guard spelled differently would restore the refusal while
    # leaving the one this test looks for in place, so no line above the
    # early return may test the mask at all.
    above = [line for line in lines[:early_return]
             if "is_active" in line and "raise" not in line
             and "getattr" not in line]
    assert above == [], (
        f"a mask test reappeared above the linear-free-surface early return: "
        f"{above}")
    guards = [line for line in lines if "is_active" in line and "None" in line]
    assert len(guards) == 1, (
        f"there must be exactly one mask guard; found {guards}")
