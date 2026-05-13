"""FV3_3D iter 627: c2l_ord2_fv3 port (D-grid → latlon winds).

Faithful port of FV3 ``c2l_ord2`` (fv_grid_utils.F90:2547-2628,
grid_type<4 branch).

Tests
-----

1. ``test_c2l_ord2_shape``.
2. ``test_c2l_ord2_zero_winds``.
3. ``test_c2l_ord2_constant_dx_dy_uniform_wind``.
4. ``test_c2l_ord2_3d_level_dim``.
5. ``test_c2l_ord2_identity_a_matrix``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import c2l_ord2_fv3


def test_c2l_ord2_shape():
    """Output shape matches cell-center grid (n_x, n_y)."""
    n_x, n_y = 6, 6
    rng = np.random.default_rng(seed=627)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    a11 = jnp.full((n_x, n_y), 0.5)
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.full((n_x, n_y), 0.5)
    ua, va = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)
    assert ua.shape == (n_x, n_y)
    assert va.shape == (n_x, n_y)


def test_c2l_ord2_zero_winds():
    """Zero D-grid winds → zero cell-center winds."""
    n_x, n_y = 4, 4
    u = jnp.zeros((n_x, n_y + 1))
    v = jnp.zeros((n_x + 1, n_y))
    dx = jnp.full((n_x, n_y + 1), 500.0)
    dy = jnp.full((n_x + 1, n_y), 500.0)
    a11 = jnp.ones((n_x, n_y))
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.ones((n_x, n_y))
    ua, va = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)
    assert jnp.allclose(ua, 0.0, atol=1e-14)
    assert jnp.allclose(va, 0.0, atol=1e-14)


def test_c2l_ord2_constant_dx_dy_uniform_wind():
    """Uniform u=U, v=V with constant dx, dy.

    FV3 averaging gives u1 = 2·U, v1 = 2·V (the FV3 ``a`` matrix
    from ``init_cubed_to_latlon`` is pre-scaled by 0.5 to
    compensate; see iter-626 ``a11 = 0.5·z22/sin_sg5``).

    For a π/2-rotation pre-scaled a-matrix
    (a11=0, a12=-0.5, a21=0.5, a22=0):
        ua = 0·2U + (-0.5)·2V = -V
        va = 0.5·2U + 0·2V = U
    """
    n_x, n_y = 4, 4
    U, V = 10.0, 5.0
    u = jnp.full((n_x, n_y + 1), U)
    v = jnp.full((n_x + 1, n_y), V)
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    # Pre-scaled rotation a-matrix (factor 0.5 absorbed)
    a11 = jnp.zeros((n_x, n_y))
    a12 = jnp.full((n_x, n_y), -0.5)
    a21 = jnp.full((n_x, n_y), 0.5)
    a22 = jnp.zeros((n_x, n_y))
    ua, va = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)
    # ua = -V, va = U (π/2 rotation)
    assert jnp.allclose(ua, -V, atol=1e-12)
    assert jnp.allclose(va, U, atol=1e-12)


def test_c2l_ord2_3d_level_dim():
    """3D D-grid input → 3D cell-center output."""
    n_x, n_y, nlev = 5, 5, 4
    rng = np.random.default_rng(seed=628)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1, nlev)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y, nlev)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    a11 = jnp.full((n_x, n_y), 0.7)
    a12 = jnp.full((n_x, n_y), 0.1)
    a21 = jnp.full((n_x, n_y), -0.1)
    a22 = jnp.full((n_x, n_y), 0.7)
    ua, va = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)
    assert ua.shape == (n_x, n_y, nlev)
    assert va.shape == (n_x, n_y, nlev)
    assert jnp.all(jnp.isfinite(ua))
    assert jnp.all(jnp.isfinite(va))


def test_c2l_ord2_identity_a_matrix():
    """Identity a-matrix → ua = u1 = vorticity-conserving avg of u.

    With dx=dy=1, u1[i, j] = 2·(u[i, j] + u[i, j+1]) / 2 = u[i, j] + u[i, j+1].
    """
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=629)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 1.0)  # constant for simplicity
    dy = jnp.full((n_x + 1, n_y), 1.0)
    a11 = jnp.ones((n_x, n_y))
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.ones((n_x, n_y))
    ua, va = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)
    # u1 = u[i, j] + u[i, j+1], v1 = v[i, j] + v[i+1, j]
    # With identity a-matrix: ua = u1, va = v1
    expected_ua = u[:, :-1] + u[:, 1:]
    expected_va = v[:-1, :] + v[1:, :]
    assert jnp.allclose(ua, expected_ua, atol=1e-12)
    assert jnp.allclose(va, expected_va, atol=1e-12)
