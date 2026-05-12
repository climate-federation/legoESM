"""FV3_3D iter 628: c2l_ord4_fv3 port (4th-order D-grid → latlon).

Faithful port of FV3 ``c2l_ord4`` (fv_grid_utils.F90:2407-2546,
grid_type<4 branch).  4-point Lagrange interpolation in interior;
2nd-order vorticity-conserving fallback at boundaries.

Tests
-----

1. ``test_c2l_ord4_shape``.
2. ``test_c2l_ord4_zero_winds``.
3. ``test_c2l_ord4_uniform_wind_constant_dx``.
4. ``test_c2l_ord4_boundary_matches_ord2``.
5. ``test_c2l_ord4_3d_level_dim``.
6. ``test_c2l_ord4_higher_order_in_interior``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import c2l_ord2_fv3, c2l_ord4_fv3


def test_c2l_ord4_shape():
    """Output shape matches cell-center grid."""
    n_x, n_y = 8, 8
    rng = np.random.default_rng(seed=628)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    a11 = jnp.full((n_x, n_y), 0.5)
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.full((n_x, n_y), 0.5)
    ua, va = c2l_ord4_fv3(u, v, dx, dy, a11, a12, a21, a22)
    assert ua.shape == (n_x, n_y)
    assert va.shape == (n_x, n_y)


def test_c2l_ord4_zero_winds():
    """Zero D-grid winds → zero output."""
    n_x, n_y = 6, 6
    u = jnp.zeros((n_x, n_y + 1))
    v = jnp.zeros((n_x + 1, n_y))
    dx = jnp.full((n_x, n_y + 1), 500.0)
    dy = jnp.full((n_x + 1, n_y), 500.0)
    a11 = jnp.ones((n_x, n_y))
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.ones((n_x, n_y))
    ua, va = c2l_ord4_fv3(u, v, dx, dy, a11, a12, a21, a22)
    assert jnp.allclose(ua, 0.0, atol=1e-14)
    assert jnp.allclose(va, 0.0, atol=1e-14)


def test_c2l_ord4_uniform_wind_constant_dx():
    """Uniform u=U, v=V → all cell centers see same projected wind."""
    n_x, n_y = 8, 8
    U, V = 7.0, 3.0
    u = jnp.full((n_x, n_y + 1), U)
    v = jnp.full((n_x + 1, n_y), V)
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    a11 = jnp.full((n_x, n_y), 0.5)
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.full((n_x, n_y), 0.5)
    ua, va = c2l_ord4_fv3(u, v, dx, dy, a11, a12, a21, a22)
    # 4-pt Lagrange on constant field: c1 + c1 + c2 + c2 = 1.125 + 1.125 - 0.125 - 0.125 = 2.0
    # So utmp = 2·U for interior; ua = 0.5·2·U = U for interior.
    # Boundary cells (c2l_ord2): u1 = 2·U, ua = 0.5·2·U = U.
    # All cells → ua = U.
    assert jnp.allclose(ua, U, atol=1e-12)
    assert jnp.allclose(va, V, atol=1e-12)


def test_c2l_ord4_boundary_matches_ord2():
    """At grid boundary, c2l_ord4 must match c2l_ord2 (FV3 fallback)."""
    n_x, n_y = 8, 8
    rng = np.random.default_rng(seed=629)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    a11 = jnp.full((n_x, n_y), 0.5)
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.full((n_x, n_y), 0.5)
    ua4, va4 = c2l_ord4_fv3(u, v, dx, dy, a11, a12, a21, a22)
    ua2, va2 = c2l_ord2_fv3(u, v, dx, dy, a11, a12, a21, a22)
    # First and last rows/cols use 2nd-order fallback
    assert jnp.allclose(ua4[0, :], ua2[0, :], atol=1e-12)
    assert jnp.allclose(ua4[-1, :], ua2[-1, :], atol=1e-12)
    assert jnp.allclose(ua4[:, 0], ua2[:, 0], atol=1e-12)
    assert jnp.allclose(ua4[:, -1], ua2[:, -1], atol=1e-12)
    assert jnp.allclose(va4[0, :], va2[0, :], atol=1e-12)
    assert jnp.allclose(va4[-1, :], va2[-1, :], atol=1e-12)


def test_c2l_ord4_3d_level_dim():
    """3D D-grid input → 3D cell-center output."""
    n_x, n_y, nlev = 6, 6, 4
    rng = np.random.default_rng(seed=630)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1, nlev)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y, nlev)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    a11 = jnp.full((n_x, n_y), 0.5)
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.full((n_x, n_y), 0.5)
    ua, va = c2l_ord4_fv3(u, v, dx, dy, a11, a12, a21, a22)
    assert ua.shape == (n_x, n_y, nlev)
    assert va.shape == (n_x, n_y, nlev)
    assert jnp.all(jnp.isfinite(ua))
    assert jnp.all(jnp.isfinite(va))


def test_c2l_ord4_higher_order_in_interior():
    """For polynomial test: ord4 should be more accurate than ord2.

    Use u(i,j) = a·j + b·j² so 4th-order interpolation captures
    quadratic exactly; 2nd-order has truncation error.  Compare
    interior values.
    """
    n_x, n_y = 12, 12
    # Build u with quadratic dependence on j (edge index along y-edge)
    j_idx = jnp.arange(n_y + 1, dtype=jnp.float64)
    u_2d = jnp.broadcast_to(0.1 * j_idx + 0.01 * j_idx ** 2, (n_x, n_y + 1))
    v_2d = jnp.zeros((n_x + 1, n_y))
    dx = jnp.full((n_x, n_y + 1), 1.0)
    dy = jnp.full((n_x + 1, n_y), 1.0)
    a11 = jnp.full((n_x, n_y), 0.5)
    a12 = jnp.zeros((n_x, n_y))
    a21 = jnp.zeros((n_x, n_y))
    a22 = jnp.full((n_x, n_y), 0.5)
    ua4, _ = c2l_ord4_fv3(u_2d, v_2d, dx, dy, a11, a12, a21, a22)
    # 4-pt Lagrange of u(j) = 0.1·j + 0.01·j² at cell-center i ∈ [1, n_y-2]:
    # utmp[i, j] = c2·(u[j-1] + u[j+2]) + c1·(u[j] + u[j+1])
    # With c1=1.125, c2=-0.125:
    # On a quadratic, this gives exact interpolation to j + 0.5.
    # So utmp[i, j] = 2 · u(j + 0.5) = 2·(0.1·(j+0.5) + 0.01·(j+0.5)²).
    # ua[i, j] = 0.5 · utmp = 0.1·(j+0.5) + 0.01·(j+0.5)²
    for jj in range(2, n_y - 2):
        expected = 0.1 * (jj + 0.5) + 0.01 * (jj + 0.5) ** 2
        # FV3 4-pt Lagrange exact on quadratic
        assert abs(float(ua4[5, jj]) - expected) < 1e-12, (
            f"j={jj}: ua4={float(ua4[5, jj])}, expected={expected}"
        )
