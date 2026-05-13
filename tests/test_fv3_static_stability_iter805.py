"""FV3_3D iter 805: static_stability_fv3 (S = −∂θ/∂p).

Pressure-coord stability parameter. Complements iter-772 N² in z-coords.

Tests
-----

1. ``test_S_stable``: θ↑ as p↓ → S > 0.
2. ``test_S_unstable``: θ↓ as p↓ → S < 0.
3. ``test_S_neutral``: θ constant → S = 0.
4. ``test_S_shapes_3d``: km → km-1.
5. ``test_S_finite``.
6. ``test_S_relation_to_N2_sign``: same sign convention as iter-772 N².
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    brunt_vaisala_squared_fv3,
    static_stability_fv3,
)


def test_S_stable():
    """Stable stratification: θ↑ as p↓ (surface→top) → S > 0."""
    # Surface→top: p decreases, θ increases
    theta = jnp.array([290.0, 295.0, 300.0, 305.0])
    p = jnp.array([100_000.0, 80_000.0, 50_000.0, 20_000.0])
    S = static_stability_fv3(theta, p)
    assert S.shape == (3,)
    assert jnp.all(S > 0.0)


def test_S_unstable():
    """Unstable: θ↓ as p↓ → S < 0."""
    theta = jnp.array([305.0, 300.0, 295.0])
    p = jnp.array([100_000.0, 80_000.0, 50_000.0])
    S = static_stability_fv3(theta, p)
    assert jnp.all(S < 0.0)


def test_S_neutral():
    """Neutral θ constant → S=0."""
    theta = jnp.full((4,), 300.0)
    p = jnp.array([100_000.0, 80_000.0, 50_000.0, 20_000.0])
    S = static_stability_fv3(theta, p)
    np.testing.assert_allclose(np.asarray(S), jnp.zeros((3,)), atol=1e-15)


def test_S_shapes_3d():
    """3-D: km → km-1."""
    rng = np.random.default_rng(seed=805)
    n_x, n_y, km = 4, 5, 20
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(n_x, n_y, km)))
    p = jnp.broadcast_to(
        jnp.linspace(100_000.0, 10_000.0, km), (n_x, n_y, km)
    )
    S = static_stability_fv3(theta, p)
    assert S.shape == (n_x, n_y, km - 1)


def test_S_finite():
    """No NaN/Inf for monotone p + realistic θ."""
    rng = np.random.default_rng(seed=806)
    km = 30
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(4, km)))
    p = jnp.broadcast_to(jnp.linspace(101_000.0, 5_000.0, km), (4, km))
    S = static_stability_fv3(theta, p)
    assert jnp.all(jnp.isfinite(S))


def test_S_relation_to_N2_sign():
    """S and N² have the same sign in stable stratification."""
    # Build hydrostatic-consistent column: θ increases with z, p decreases
    z = jnp.linspace(0.0, 15_000.0, 16)
    theta = 290.0 + 0.005 * z  # 5 K/km stable
    p = 101_000.0 * jnp.exp(-z / 8000.0)
    q = jnp.zeros_like(theta)
    n_sq = brunt_vaisala_squared_fv3(theta, q, z)
    S = static_stability_fv3(theta, p)
    # Both should be positive (stable atmosphere)
    assert jnp.all(n_sq > 0.0)
    assert jnp.all(S > 0.0)
