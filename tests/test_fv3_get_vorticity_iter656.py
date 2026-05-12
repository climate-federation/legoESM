"""FV3_3D iter 656: get_vorticity_fv3 port.

Faithful JAX port of FV3 ``get_vorticity`` (tools/test_cases.F90:
4034-4065).  Cell-center vorticity from D-grid winds via
line-integral / cell-area form.

Tests
-----

1. ``test_vorticity_shape``.
2. ``test_vorticity_zero_winds``.
3. ``test_vorticity_uniform_winds``.
4. ``test_vorticity_solid_body_rotation``.
5. ``test_vorticity_3d_level_dim``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import get_vorticity_fv3


def test_vorticity_shape():
    """Output shape (n_x, n_y) from D-grid winds."""
    n_x, n_y = 5, 7
    rng = np.random.default_rng(seed=656)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    rarea = jnp.full((n_x, n_y), 1.0e-6)
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert vort.shape == (n_x, n_y)


def test_vorticity_zero_winds():
    """Zero winds → zero vorticity."""
    n_x, n_y = 4, 4
    u = jnp.zeros((n_x, n_y + 1))
    v = jnp.zeros((n_x + 1, n_y))
    dx = jnp.full((n_x, n_y + 1), 500.0)
    dy = jnp.full((n_x + 1, n_y), 500.0)
    rarea = jnp.full((n_x, n_y), 4.0e-6)
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert jnp.allclose(vort, 0.0, atol=1e-14)


def test_vorticity_uniform_winds():
    """Uniform u=U, v=V → vorticity = 0 (no rotation)."""
    n_x, n_y = 4, 4
    U, V = 10.0, 5.0
    u = jnp.full((n_x, n_y + 1), U)
    v = jnp.full((n_x + 1, n_y), V)
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    rarea = jnp.full((n_x, n_y), 1.0e-6)
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert jnp.allclose(vort, 0.0, atol=1e-12)


def test_vorticity_solid_body_rotation():
    """Solid-body rotation u=-Ω·y, v=Ω·x in Cartesian gives vort=2Ω.

    Build artificial D-grid winds: u[i, j] depends on j (linear in y),
    v[i, j] depends on i (linear in x).  Vorticity = -du/dy + dv/dx = 2Ω.
    """
    n_x, n_y = 6, 6
    Omega = 0.001
    # Build u(j) = -Ω·j (decreasing with j)
    j_edges = jnp.arange(n_y + 1, dtype=jnp.float64)
    u_2d = jnp.broadcast_to(-Omega * j_edges, (n_x, n_y + 1))
    # v(i) = Ω·i (increasing with i)
    i_edges = jnp.arange(n_x + 1, dtype=jnp.float64)
    v_2d = jnp.broadcast_to((Omega * i_edges)[:, None], (n_x + 1, n_y))
    # Use dx = dy = 1, rarea = 1
    dx = jnp.ones((n_x, n_y + 1))
    dy = jnp.ones((n_x + 1, n_y))
    rarea = jnp.ones((n_x, n_y))
    vort = get_vorticity_fv3(u_2d, v_2d, dx, dy, rarea)
    # vort = (u[i, j] - u[i, j+1]) + (v[i+1, j] - v[i, j])
    #      = (-Ω·j - (-Ω·(j+1))) + (Ω·(i+1) - Ω·i)
    #      = Ω + Ω = 2·Ω
    assert jnp.allclose(vort, 2.0 * Omega, atol=1e-12)


def test_vorticity_3d_level_dim():
    """3D D-grid input → 3D vorticity output."""
    n_x, n_y, nlev = 5, 5, 4
    rng = np.random.default_rng(seed=657)
    u = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1, nlev)))
    v = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y, nlev)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    rarea = jnp.full((n_x, n_y), 1.0e-6)
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    assert vort.shape == (n_x, n_y, nlev)
    assert jnp.all(jnp.isfinite(vort))
