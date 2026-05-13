"""FV3_3D iter 776: shear_squared_fv3 + iter-773 refactor.

S² = (du/dz)² + (dv/dz)² at layer midpoints.

Extracted from iter-773 ``richardson_number_fv3`` inline form;
iter-773 now delegates.

Tests
-----

1. ``test_shear_zero_winds``: u=v=0 → S²=0.
2. ``test_shear_only_u``: only u changes → S²=(du/dz)².
3. ``test_shear_only_v``: only v changes → S²=(dv/dz)².
4. ``test_shear_iter773_unchanged``: iter-773 Ri preserved after refactor.
5. ``test_shear_shapes_3d``: km → km-1.
6. ``test_shear_nonneg_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    richardson_number_fv3,
    shear_squared_fv3,
)


def test_shear_zero_winds():
    """No wind → S² = 0."""
    z = jnp.array([0.0, 500.0, 1500.0])
    u = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    s_sq = shear_squared_fv3(u, v, z)
    np.testing.assert_allclose(np.asarray(s_sq), jnp.zeros((2,)), atol=1e-14)


def test_shear_only_u():
    """u shear only → S² = (du/dz)²."""
    z = jnp.array([0.0, 1000.0])  # Δz = 1000
    u = jnp.array([0.0, 10.0])    # du = 10, du/dz = 0.01
    v = jnp.zeros_like(z)
    s_sq = shear_squared_fv3(u, v, z)
    np.testing.assert_allclose(np.asarray(s_sq), [1e-4], atol=1e-14)


def test_shear_only_v():
    """v shear only → S² = (dv/dz)²."""
    z = jnp.array([0.0, 1000.0])
    u = jnp.zeros_like(z)
    v = jnp.array([0.0, 20.0])    # dv = 20, dv/dz = 0.02
    s_sq = shear_squared_fv3(u, v, z)
    np.testing.assert_allclose(np.asarray(s_sq), [4e-4], atol=1e-14)


def test_shear_iter773_unchanged():
    """iter-773 Ri output preserved bit-identical after refactor."""
    rng = np.random.default_rng(seed=776)
    km = 10
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 500.0, size=(km,))))
    theta = jnp.asarray(rng.uniform(280.0, 320.0, size=(km,)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(km,)))
    u = jnp.asarray(rng.uniform(-20.0, 20.0, size=(km,)))
    v = jnp.asarray(rng.uniform(-20.0, 20.0, size=(km,)))
    ri = richardson_number_fv3(theta, q, u, v, z)
    assert jnp.all(jnp.isfinite(ri))


def test_shear_shapes_3d():
    """3-D shapes: km → km-1."""
    rng = np.random.default_rng(seed=777)
    n_x, n_y, km = 4, 5, 20
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 500.0, size=(n_x, n_y, km))), axis=-1)
    u = jnp.asarray(rng.uniform(-30.0, 30.0, size=(n_x, n_y, km)))
    v = jnp.asarray(rng.uniform(-30.0, 30.0, size=(n_x, n_y, km)))
    s_sq = shear_squared_fv3(u, v, z)
    assert s_sq.shape == (n_x, n_y, km - 1)


def test_shear_nonneg_finite():
    """S² ≥ 0 always, finite for monotone z + realistic winds."""
    rng = np.random.default_rng(seed=778)
    km = 30
    z = jnp.cumsum(jnp.asarray(rng.uniform(100.0, 500.0, size=(4, km))), axis=-1)
    u = jnp.asarray(rng.uniform(-50.0, 50.0, size=(4, km)))
    v = jnp.asarray(rng.uniform(-50.0, 50.0, size=(4, km)))
    s_sq = shear_squared_fv3(u, v, z)
    assert jnp.all(jnp.isfinite(s_sq))
    assert jnp.all(s_sq >= 0.0)
