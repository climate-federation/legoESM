"""FV3_3D iter 791: ageostrophic_wind_fv3.

V_a = V − V_g.

Composes iter-790 ``geostrophic_wind_fv3``.

Tests
-----

1. ``test_va_pure_geostrophic``: V=V_g → V_a=0.
2. ``test_va_simple_difference``: V_a equals direct subtraction.
3. ``test_va_decomposition_identity``: V_g + V_a = V exactly.
4. ``test_va_composes_iter790``: full (V, ∇Φ, f) → V_a chain.
5. ``test_va_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    ageostrophic_wind_fv3,
    geostrophic_wind_fv3,
)


def test_va_pure_geostrophic():
    """V=V_g → V_a=0."""
    u = jnp.array([10.0, 5.0, -3.0])
    v = jnp.array([0.0, -2.0, 8.0])
    u_a, v_a = ageostrophic_wind_fv3(u, v, u, v)
    np.testing.assert_allclose(np.asarray(u_a), jnp.zeros((3,)), atol=1e-15)
    np.testing.assert_allclose(np.asarray(v_a), jnp.zeros((3,)), atol=1e-15)


def test_va_simple_difference():
    """V_a = V − V_g directly."""
    u = jnp.array([10.0, 5.0])
    v = jnp.array([3.0, -1.0])
    u_g = jnp.array([8.0, 6.0])
    v_g = jnp.array([2.0, 1.0])
    u_a, v_a = ageostrophic_wind_fv3(u, v, u_g, v_g)
    np.testing.assert_allclose(np.asarray(u_a), [2.0, -1.0], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v_a), [1.0, -2.0], rtol=1e-12)


def test_va_decomposition_identity():
    """V_g + V_a = V exactly."""
    rng = np.random.default_rng(seed=791)
    u = jnp.asarray(rng.uniform(-30.0, 30.0, size=(20,)))
    v = jnp.asarray(rng.uniform(-30.0, 30.0, size=(20,)))
    u_g = jnp.asarray(rng.uniform(-30.0, 30.0, size=(20,)))
    v_g = jnp.asarray(rng.uniform(-30.0, 30.0, size=(20,)))
    u_a, v_a = ageostrophic_wind_fv3(u, v, u_g, v_g)
    np.testing.assert_allclose(np.asarray(u_g + u_a), np.asarray(u), atol=1e-14)
    np.testing.assert_allclose(np.asarray(v_g + v_a), np.asarray(v), atol=1e-14)


def test_va_composes_iter790():
    """Full chain: (V, ∇Φ, f) → V_g → V_a."""
    u = jnp.array([15.0])
    v = jnp.array([5.0])
    dphi_dx = jnp.array([1e-3])
    dphi_dy = jnp.array([-1e-3])
    f = jnp.array([1.0e-4])
    u_g, v_g = geostrophic_wind_fv3(dphi_dx, dphi_dy, f)
    # u_g = -(-1e-3)/1e-4 = +10; v_g = 1e-3/1e-4 = +10
    u_a, v_a = ageostrophic_wind_fv3(u, v, u_g, v_g)
    # u_a = 15 - 10 = +5; v_a = 5 - 10 = -5
    np.testing.assert_allclose(np.asarray(u_a), [5.0], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v_a), [-5.0], rtol=1e-12)


def test_va_shapes_3d_finite():
    """3-D shapes preserved, finite, both components."""
    rng = np.random.default_rng(seed=792)
    n_x, n_y, km = 4, 5, 20
    u = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n_x, n_y, km)))
    v = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n_x, n_y, km)))
    u_g = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n_x, n_y, km)))
    v_g = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n_x, n_y, km)))
    u_a, v_a = ageostrophic_wind_fv3(u, v, u_g, v_g)
    assert u_a.shape == (n_x, n_y, km)
    assert v_a.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(u_a))
    assert jnp.all(jnp.isfinite(v_a))
