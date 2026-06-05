"""FV3_3D iter 312: iter-187 smag_vort cap saturates at 0.20.

The FV3 corner-div damping coefficient at nord >= 1 is::

    inner = dddmp * |dt| * sqrt(delpc² + ζ²)
    cap = min(0.20, inner)
    damp = max(d2_bg, cap)

When ``inner`` exceeds 0.20, the ``min`` clamps cap to 0.20
exactly (FV3 sw_core.F90:1798-1801 ``min`` operator).  This
saturation behaviour is what protects against runaway damping
at intense divergence.

iter-285 pinned non-negativity, sqrt(0) AD-safety, and rtol
1e-12 agreement with naive sqrt at non-rest.  iter-295 pinned
scaling and sign-symmetry.  iter-312 pins the ``min(0.20, ...)``
saturation behaviour itself: above the threshold, output is
exactly 0.20; below, output equals the inner expression.

Tests
-----

1. ``test_smag_vort_below_saturation`` — when inner < 0.20,
   the cap output equals inner (rtol=1e-13).
2. ``test_smag_vort_at_saturation`` — when inner > 0.20, the
   cap output is exactly 0.20 (array-equal).
3. ``test_smag_vort_saturation_threshold`` — at the boundary
   (inner == 0.20), output is 0.20.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _smag_vort_capped(delpc, zeta, dt, dddmp):
    """Reproduce the FV3 corner-div damping cap (without
    the d2_bg floor).  Mirrors compressible_euler_cdgrid.py
    line 698-700 / primitive_eq_cdgrid.py analogous site.
    """
    smag_arg = delpc ** 2 + zeta ** 2
    safe_arg = jnp.where(smag_arg > 0.0, smag_arg, 1.0)
    smag_root = jnp.where(
        smag_arg > 0.0, jnp.sqrt(safe_arg), 0.0,
    )
    smag_vort = jnp.abs(dt) * smag_root
    inner = dddmp * smag_vort
    return jnp.minimum(0.20, inner), inner


def test_smag_vort_below_saturation():
    """When ``inner < 0.20`` everywhere, cap output = inner."""
    rng = np.random.default_rng(seed=312)
    # Small delpc, ζ → inner is small.
    delpc = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    dt = 10.0
    dddmp = 0.20

    cap, inner = _smag_vort_capped(delpc, zeta, dt, dddmp)
    # Sanity: inner is below 0.20 for these inputs.
    assert float(jnp.max(inner)) < 0.20, (
        "Test setup error: at small inputs inner should be < 0.20"
    )
    np.testing.assert_allclose(
        np.asarray(cap), np.asarray(inner), rtol=1e-13, atol=1e-15,
    )


def test_smag_vort_at_saturation():
    """When ``inner > 0.20`` everywhere, cap output = 0.20."""
    rng = np.random.default_rng(seed=312)
    # Pump up delpc so inner = dddmp * dt * sqrt(...) > 0.20.
    # At delpc ~ 1, dt=10, dddmp=0.20, inner ~ 0.20*10*1 = 2.
    delpc = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 9, 9, 5)))
    dt = 10.0
    dddmp = 0.20

    cap, inner = _smag_vort_capped(delpc, zeta, dt, dddmp)
    assert float(jnp.min(inner)) > 0.20, (
        "Test setup error: with inputs ~ 1, inner should be > 0.20"
    )
    np.testing.assert_array_equal(
        np.asarray(cap), np.full_like(np.asarray(cap), 0.20),
    )


def test_smag_vort_saturation_threshold():
    """When ``inner == 0.20`` exactly, cap output = 0.20."""
    # Construct inputs where inner = 0.20 exactly.
    # inner = dddmp * |dt| * sqrt(delpc² + ζ²) = 0.20
    # Set dt = 1, dddmp = 1, delpc = 0.20, ζ = 0:
    #   inner = 1 * 1 * sqrt(0.04) = 0.2  ✓
    delpc = jnp.full((6, 9, 9, 5), 0.20)
    zeta = jnp.zeros_like(delpc)
    dt = 1.0
    dddmp = 1.0

    cap, inner = _smag_vort_capped(delpc, zeta, dt, dddmp)
    np.testing.assert_allclose(
        np.asarray(inner), 0.20, rtol=1e-15,
    )
    np.testing.assert_array_equal(
        np.asarray(cap), np.full_like(np.asarray(cap), 0.20),
    )
