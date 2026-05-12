"""FV3_3D iter 773: richardson_number_fv3 (gradient Ri).

Ri = N² / S² where S² = (du/dz)² + (dv/dz)².

Composes iter-772 ``brunt_vaisala_squared_fv3``.

Tests
-----

1. ``test_ri_calm_air_floor``: zero shear → very large Ri.
2. ``test_ri_strong_shear_neutral``: strong shear + neutral → Ri≈0.
3. ``test_ri_strong_stable_weak_shear``: stable + weak shear → Ri>1.
4. ``test_ri_composes_with_n2``: Ri = N²/S² explicitly.
5. ``test_ri_shapes_3d``: km → km-1.
6. ``test_ri_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    brunt_vaisala_squared_fv3,
    richardson_number_fv3,
)


def test_ri_calm_air_floor():
    """Zero shear (calm) → Ri large (clamped by shear floor)."""
    z = jnp.array([0.0, 1000.0, 2000.0])
    theta = jnp.array([290.0, 295.0, 300.0])  # stable
    q = jnp.zeros_like(theta)
    u = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    ri = richardson_number_fv3(theta, q, u, v, z)
    # With shear_floor=1e-12 and N²>0, Ri is huge but finite
    assert jnp.all(ri > 1e6)
    assert jnp.all(jnp.isfinite(ri))


def test_ri_strong_shear_neutral():
    """Strong shear + neutral stratification → Ri ≈ 0."""
    z = jnp.array([0.0, 500.0, 1000.0])
    theta = jnp.full((3,), 300.0)  # neutral
    q = jnp.zeros_like(theta)
    u = jnp.array([0.0, 20.0, 40.0])  # strong shear
    v = jnp.zeros_like(theta)
    ri = richardson_number_fv3(theta, q, u, v, z)
    np.testing.assert_allclose(np.asarray(ri), jnp.zeros((2,)), atol=1e-10)


def test_ri_strong_stable_weak_shear():
    """Strong stable stratification + weak shear → Ri > 1."""
    z = jnp.array([0.0, 1000.0, 2000.0])
    theta = jnp.array([280.0, 295.0, 310.0])  # very stable
    q = jnp.zeros_like(theta)
    u = jnp.array([0.0, 0.5, 1.0])  # weak shear (1 m/s per 2 km)
    v = jnp.zeros_like(theta)
    ri = richardson_number_fv3(theta, q, u, v, z)
    assert jnp.all(ri > 1.0)


def test_ri_composes_with_n2():
    """Ri = N²/S² explicit check."""
    rng = np.random.default_rng(seed=773)
    km = 10
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 500.0, size=(km,))))
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(km,)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(km,)))
    u = jnp.asarray(rng.uniform(-20.0, 20.0, size=(km,)))
    v = jnp.asarray(rng.uniform(-20.0, 20.0, size=(km,)))
    ri = richardson_number_fv3(theta, q, u, v, z)
    # Reference: N² from iter-772, S² inline
    n_sq = brunt_vaisala_squared_fv3(theta, q, z)
    du = u[1:] - u[:-1]
    dv = v[1:] - v[:-1]
    dz = z[1:] - z[:-1]
    s_sq = (du / dz) ** 2 + (dv / dz) ** 2
    ri_ref = n_sq / jnp.maximum(s_sq, 1e-12)
    np.testing.assert_allclose(np.asarray(ri), np.asarray(ri_ref), atol=1e-12)


def test_ri_shapes_3d():
    """3-D shapes: km → km-1."""
    rng = np.random.default_rng(seed=774)
    n_x, n_y, km = 4, 5, 20
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 500.0, size=(n_x, n_y, km))), axis=-1)
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    u = jnp.asarray(rng.uniform(-30.0, 30.0, size=(n_x, n_y, km)))
    v = jnp.asarray(rng.uniform(-30.0, 30.0, size=(n_x, n_y, km)))
    ri = richardson_number_fv3(theta, q, u, v, z)
    assert ri.shape == (n_x, n_y, km - 1)


def test_ri_finite():
    """No NaN/Inf for monotone z + realistic fields."""
    rng = np.random.default_rng(seed=775)
    km = 30
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 500.0, size=(4, km))), axis=-1)
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(4, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(4, km)))
    u = jnp.asarray(rng.uniform(-50.0, 50.0, size=(4, km)))
    v = jnp.asarray(rng.uniform(-50.0, 50.0, size=(4, km)))
    ri = richardson_number_fv3(theta, q, u, v, z)
    assert jnp.all(jnp.isfinite(ri))
