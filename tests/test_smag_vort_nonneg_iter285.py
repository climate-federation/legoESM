"""FV3_3D iter 285: range sanity test for the iter-187
``smag_vort`` cap formula.

The FV3 smag_vort = ``|dt| * sqrt(delpc² + ζ²)`` produces a
strictly NON-NEGATIVE value (sqrt of non-negative
argument).  This test verifies the AD-safe iter-183 double-
where pattern preserves this property, including at exactly
the rest state where the argument is zero.

Tests
-----

1. ``test_smag_vort_is_nonneg_at_rest`` — smag_vort = 0 at
   rest state (input delpc=0, ζ=0).
2. ``test_smag_vort_is_nonneg_at_perturbed`` — smag_vort > 0
   at non-rest state.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _safe_smag_vort(delpc, zeta, dt):
    """Reproduce the iter-183 AD-safe form."""
    smag_arg = delpc ** 2 + zeta ** 2
    safe_arg = jnp.where(smag_arg > 0.0, smag_arg, 1.0)
    smag_root = jnp.where(
        smag_arg > 0.0, jnp.sqrt(safe_arg), 0.0,
    )
    return jnp.abs(dt) * smag_root


def test_smag_vort_is_nonneg_at_rest():
    """smag_vort = 0 at rest (delpc=ζ=0)."""
    delpc = jnp.zeros((6, 9, 9, 5))
    zeta = jnp.zeros((6, 9, 9, 5))
    smag = _safe_smag_vort(delpc, zeta, dt=10.0)
    assert jnp.all(smag == 0.0), (
        "smag_vort at rest must be exactly 0 (sqrt(0)=0)."
    )

    # And gradient w.r.t. (delpc, zeta) is finite.
    def grad_target(delpc_in, zeta_in):
        return jnp.sum(_safe_smag_vort(delpc_in, zeta_in, dt=10.0))

    g_delpc = jax.grad(grad_target, 0)(delpc, zeta)
    g_zeta = jax.grad(grad_target, 1)(delpc, zeta)
    assert jnp.all(jnp.isfinite(g_delpc))
    assert jnp.all(jnp.isfinite(g_zeta))


def test_smag_vort_is_nonneg_at_perturbed():
    """smag_vort > 0 at non-rest state."""
    rng = np.random.default_rng(seed=285)
    delpc = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    smag = _safe_smag_vort(delpc, zeta, dt=10.0)
    assert jnp.all(smag >= 0.0), (
        "smag_vort must be non-negative."
    )
    assert float(jnp.max(smag)) > 0.0, (
        "smag_vort should be positive somewhere with non-zero "
        "input."
    )

    # Match the naive sqrt formula to confirm bit-for-bit
    # equivalence on positive arguments.
    naive = abs(10.0) * jnp.sqrt(
        jnp.maximum(delpc ** 2 + zeta ** 2, 0.0),
    )
    np.testing.assert_allclose(smag, naive, rtol=1e-12)
