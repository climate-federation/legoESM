"""FV3_3D iter 654: atoc_vort_on port.

Faithful JAX port of FV3 ``atoc`` (tools/test_cases.F90:7965-8112,
VORT_ON branch, no ALT_INTERP).  Circulation-conserving A→C grid.

Tests
-----

1. ``test_atoc_shape``.
2. ``test_atoc_zero_winds``.
3. ``test_atoc_uniform_winds_interior``.
4. ``test_atoc_boundary_zero``.
5. ``test_atoc_simple_average_when_dxa_uniform``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import atoc_vort_on


def test_atoc_shape():
    """Output uout (n_x+1, n_y); vout (n_x, n_y+1)."""
    n_x, n_y = 5, 7
    rng = np.random.default_rng(seed=654)
    uin = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y)))
    vin = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y)))
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atoc_vort_on(uin, vin, dxa, dya)
    assert uout.shape == (n_x + 1, n_y)
    assert vout.shape == (n_x, n_y + 1)


def test_atoc_zero_winds():
    """Zero A-grid winds → zero C-grid output."""
    n_x, n_y = 4, 4
    uin = jnp.zeros((n_x, n_y))
    vin = jnp.zeros((n_x, n_y))
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atoc_vort_on(uin, vin, dxa, dya)
    assert jnp.allclose(uout, 0.0, atol=1e-14)
    assert jnp.allclose(vout, 0.0, atol=1e-14)


def test_atoc_uniform_winds_interior():
    """Uniform u=U, v=V → interior C-grid = U, V."""
    n_x, n_y = 4, 4
    U, V = 7.0, 3.0
    uin = jnp.full((n_x, n_y), U)
    vin = jnp.full((n_x, n_y), V)
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atoc_vort_on(uin, vin, dxa, dya)
    # Interior: u_out[i ∈ [1, n_x-1], j]
    assert jnp.allclose(uout[1:n_x, :], U, atol=1e-12)
    assert jnp.allclose(vout[:, 1:n_y], V, atol=1e-12)


def test_atoc_boundary_zero():
    """Boundary edges (uout[0, :], uout[-1, :]) remain 0 (FV3 fill-halo
    later)."""
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=655)
    uin = jnp.asarray(rng.uniform(1, 10, size=(n_x, n_y)))
    vin = jnp.asarray(rng.uniform(1, 10, size=(n_x, n_y)))
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atoc_vort_on(uin, vin, dxa, dya)
    assert jnp.allclose(uout[0, :], 0.0, atol=1e-14)
    assert jnp.allclose(uout[-1, :], 0.0, atol=1e-14)
    assert jnp.allclose(vout[:, 0], 0.0, atol=1e-14)
    assert jnp.allclose(vout[:, -1], 0.0, atol=1e-14)


def test_atoc_simple_average_when_dxa_uniform():
    """With dxa=const, interior uout[i, j] = 0.5·(uin[i-1, j] + uin[i, j])."""
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=656)
    uin = jnp.asarray(rng.uniform(-10, 10, size=(n_x, n_y)))
    vin = jnp.asarray(rng.uniform(-10, 10, size=(n_x, n_y)))
    dxa = jnp.full((n_x, n_y), 1.5)
    dya = jnp.full((n_x, n_y), 1.5)
    uout, vout = atoc_vort_on(uin, vin, dxa, dya)
    expected_uout_interior = 0.5 * (uin[:-1, :] + uin[1:, :])  # (n_x-1, n_y)
    expected_vout_interior = 0.5 * (vin[:, :-1] + vin[:, 1:])  # (n_x, n_y-1)
    assert jnp.allclose(uout[1:n_x, :], expected_uout_interior, atol=1e-12)
    assert jnp.allclose(vout[:, 1:n_y], expected_vout_interior, atol=1e-12)
