"""FV3_3D iter 313: corner-div damp ``d2_bg`` floor saturation.

The full FV3 corner-div damping coefficient at nord >= 1 is::

    inner = dddmp * |dt| * sqrt(delpc² + ζ²)
    cap = min(0.20, inner)
    damp_corner = da_min_c * max(d2_bg, cap)

iter-312 pinned the ``min(0.20, ...)`` ceiling.  iter-313 pins
the orthogonal ``max(d2_bg, ...)`` floor: when the cap is small
(small delpc / ζ at small dt), d2_bg dominates and provides a
constant background damping; when the cap exceeds d2_bg, it
takes over.

This is the FV3 sw_core.F90:1801-1804 ``max(d2_bg, ...)``
operator — the d2_bg floor ensures a non-zero damping
coefficient even at quiescent flow, providing baseline
numerical stability.

Tests
-----

1. ``test_corner_div_damp_below_floor`` — when cap < d2_bg,
   damp = d2_bg exactly.
2. ``test_corner_div_damp_above_floor`` — when cap > d2_bg,
   damp = cap.
3. ``test_corner_div_damp_at_floor_threshold`` — at boundary,
   damp = d2_bg = cap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _full_corner_div_damp_coeff(delpc, zeta, dt, dddmp, d2_bg):
    """Reproduce the full FV3 corner-div damp coefficient
    excluding the area normalization.

    Returns (damp_normalized, cap_inner, inner_unclamped).
    """
    smag_arg = delpc ** 2 + zeta ** 2
    safe_arg = jnp.where(smag_arg > 0.0, smag_arg, 1.0)
    smag_root = jnp.where(
        smag_arg > 0.0, jnp.sqrt(safe_arg), 0.0,
    )
    smag_vort = jnp.abs(dt) * smag_root
    inner = dddmp * smag_vort
    cap = jnp.minimum(0.20, inner)
    damp = jnp.maximum(d2_bg, cap)
    return damp, cap, inner


def test_corner_div_damp_below_floor():
    """When cap < d2_bg, damp = d2_bg exactly."""
    rng = np.random.default_rng(seed=313)
    # Tiny delpc, ζ → cap is tiny.
    delpc = jnp.asarray(rng.uniform(-1e-6, 1e-6, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(-1e-6, 1e-6, size=(6, 9, 9, 5)))
    dt = 10.0
    dddmp = 0.20
    d2_bg = 0.005

    damp, cap, _ = _full_corner_div_damp_coeff(
        delpc, zeta, dt, dddmp, d2_bg,
    )
    assert float(jnp.max(cap)) < d2_bg, (
        "Test setup error: at tiny inputs cap should be < d2_bg"
    )
    np.testing.assert_array_equal(
        np.asarray(damp), np.full_like(np.asarray(damp), d2_bg),
    )


def test_corner_div_damp_above_floor():
    """When cap > d2_bg everywhere, damp = cap (rtol=1e-13)."""
    rng = np.random.default_rng(seed=313)
    # Moderate delpc, ζ → cap > d2_bg, but cap < 0.20 to avoid
    # the iter-312 ceiling activation.
    # cap = dddmp * dt * sqrt(...) = 0.20 * 10 * sqrt(d²+z²) = 2*sqrt(...)
    # Want cap ∈ (d2_bg=1e-6, 0.20):
    #   sqrt > 1e-6/2 = 5e-7  AND  sqrt < 0.10
    # Use delpc, ζ ~ 1e-4 → sqrt ~ 1e-4 → cap ~ 2e-4.  Good.
    delpc = jnp.asarray(rng.uniform(0.5e-4, 1.5e-4, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(0.5e-4, 1.5e-4, size=(6, 9, 9, 5)))
    dt = 10.0
    dddmp = 0.20
    d2_bg = 1e-6  # tiny floor so cap dominates

    damp, cap, _ = _full_corner_div_damp_coeff(
        delpc, zeta, dt, dddmp, d2_bg,
    )
    assert float(jnp.min(cap)) > d2_bg, (
        "Test setup error: cap should exceed d2_bg here"
    )
    assert float(jnp.max(cap)) < 0.20, (
        "Test setup error: cap should not hit the 0.20 ceiling"
    )
    np.testing.assert_allclose(
        np.asarray(damp), np.asarray(cap), rtol=1e-13, atol=1e-15,
    )


def test_corner_div_damp_at_floor_threshold():
    """At cap == d2_bg exactly, damp = d2_bg (max picks either)."""
    # Construct so that inner = d2_bg = 0.05 exactly.
    # inner = dddmp * |dt| * sqrt(delpc² + ζ²)
    # Set dddmp=1, dt=1, delpc=0.05, ζ=0:
    #   inner = 0.05  ✓  cap = min(0.20, 0.05) = 0.05  ✓
    delpc = jnp.full((6, 9, 9, 5), 0.05)
    zeta = jnp.zeros_like(delpc)
    dt = 1.0
    dddmp = 1.0
    d2_bg = 0.05

    damp, cap, _ = _full_corner_div_damp_coeff(
        delpc, zeta, dt, dddmp, d2_bg,
    )
    np.testing.assert_allclose(np.asarray(cap), 0.05, rtol=1e-15)
    np.testing.assert_array_equal(
        np.asarray(damp), np.full_like(np.asarray(damp), d2_bg),
    )
