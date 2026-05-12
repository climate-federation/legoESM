"""FV3_3D iter 593: FV3 iord=10 PPM limiter utility.

Tests
-----

1. ``test_hord10_returns_finite``.
2. ``test_hord10_flat_field_zeros`` — uniform q → dm ~ 0 → bl, br = 0.
3. ``test_hord10_preserves_boundary`` — first/last 2 cells unchanged.
4. ``test_hord10_smooth_field_passes_through`` — no new extremum →
   bl, br unchanged.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import apply_hord10_limiter


def test_hord10_returns_finite():
    rng = np.random.default_rng(seed=593)
    n = 16
    bl = jnp.asarray(rng.uniform(-1, 1, size=(n,)))
    br = jnp.asarray(rng.uniform(-1, 1, size=(n,)))
    dm = jnp.asarray(rng.uniform(-0.5, 0.5, size=(n,)))
    q = jnp.asarray(rng.uniform(0, 10, size=(n,)))
    bl_out, br_out = apply_hord10_limiter(bl, br, dm, q)
    assert jnp.all(jnp.isfinite(bl_out))
    assert jnp.all(jnp.isfinite(br_out))


def test_hord10_flat_field_zeros():
    """Uniform q → dm = 0 → interior bl, br forced to 0."""
    n = 16
    bl = jnp.ones((n,)) * 0.5  # arbitrary non-zero
    br = jnp.ones((n,)) * 0.3
    dm = jnp.zeros((n,))  # flat → near_zero condition triggers
    q = jnp.ones((n,)) * 5.0  # uniform → dq = 0
    bl_out, br_out = apply_hord10_limiter(bl, br, dm, q)
    # Interior cells [2:-2] should be zeroed
    assert jnp.all(jnp.abs(bl_out[2:-2]) < 1e-12)
    assert jnp.all(jnp.abs(br_out[2:-2]) < 1e-12)


def test_hord10_preserves_boundary():
    """First/last 2 cells unchanged (caller handles halos)."""
    n = 10
    rng = np.random.default_rng(seed=594)
    bl_orig = jnp.asarray(rng.uniform(-1, 1, size=(n,)))
    br_orig = jnp.asarray(rng.uniform(-1, 1, size=(n,)))
    dm = jnp.asarray(rng.uniform(-0.5, 0.5, size=(n,)))
    q = jnp.asarray(rng.uniform(0, 10, size=(n,)))
    bl_out, br_out = apply_hord10_limiter(bl_orig, br_orig, dm, q)
    # Boundary cells preserved
    for i in [0, 1, -2, -1]:
        assert float(bl_out[i]) == float(bl_orig[i]), (
            f"bl boundary cell {i} should be preserved"
        )
        assert float(br_out[i]) == float(br_orig[i]), (
            f"br boundary cell {i} should be preserved"
        )


def test_hord10_smooth_field_passes_through():
    """A smooth field where |3·(bl+br)| < |bl-br| (no new extremum)
    → bl, br pass through unchanged."""
    n = 10
    # Construct bl, br such that 3·(bl+br) is small relative to bl-br.
    bl = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    br = -bl  # opposite signs → bl+br = 0, no new extremum condition
    dm = jnp.full((n,), 0.5)  # non-flat
    q = jnp.arange(n, dtype=jnp.float64)  # monotone
    bl_out, br_out = apply_hord10_limiter(bl, br, dm, q)
    # No new extremum: bl, br should be unchanged in interior
    diff_bl = float(jnp.abs(bl_out[2:-2] - bl[2:-2]).max())
    diff_br = float(jnp.abs(br_out[2:-2] - br[2:-2]).max())
    assert diff_bl < 1e-12, f"smooth bl should be unchanged; diff={diff_bl}"
    assert diff_br < 1e-12, f"smooth br should be unchanged; diff={diff_br}"
