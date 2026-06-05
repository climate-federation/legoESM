"""FV3_3D iter 652: dtoa_vort_on port.

Faithful JAX port of FV3 ``dtoa`` (tools/test_cases.F90:7896-7955,
VORT_ON branch).  Circulation-conserving D-grid → A-grid winds.

Tests
-----

1. ``test_dtoa_shape``.
2. ``test_dtoa_zero_winds``.
3. ``test_dtoa_uniform_winds``.
4. ``test_dtoa_circulation_conserving``.
5. ``test_dtoa_batched``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import dtoa_vort_on


def test_dtoa_shape():
    """Output shape (n_x, n_y) from D-grid (n_x, n_y+1), (n_x+1, n_y)."""
    n_x, n_y = 5, 7
    rng = np.random.default_rng(seed=652)
    uin = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    vin = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = dtoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert uout.shape == (n_x, n_y)
    assert vout.shape == (n_x, n_y)


def test_dtoa_zero_winds():
    """Zero D-grid winds → zero A-grid output."""
    n_x, n_y = 4, 4
    uin = jnp.zeros((n_x, n_y + 1))
    vin = jnp.zeros((n_x + 1, n_y))
    dx = jnp.full((n_x, n_y + 1), 500.0)
    dy = jnp.full((n_x + 1, n_y), 500.0)
    dxa = jnp.full((n_x, n_y), 500.0)
    dya = jnp.full((n_x, n_y), 500.0)
    uout, vout = dtoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert jnp.allclose(uout, 0.0, atol=1e-14)
    assert jnp.allclose(vout, 0.0, atol=1e-14)


def test_dtoa_uniform_winds():
    """Uniform u=U, v=V with uniform dx, dy → uout=U, vout=V."""
    n_x, n_y = 4, 4
    U, V = 10.0, 5.0
    uin = jnp.full((n_x, n_y + 1), U)
    vin = jnp.full((n_x + 1, n_y), V)
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = dtoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert jnp.allclose(uout, U, atol=1e-12)
    assert jnp.allclose(vout, V, atol=1e-12)


def test_dtoa_circulation_conserving():
    """Circulation Σ u·dx over a closed path is preserved.

    Take a 2D rectangle of cells; sum u·dx around the perimeter (D-grid)
    equals same sum reconstructed from A-grid output (within numeric
    precision).  Simplified test: just check that the formula linearly
    averages the two adjacent D-grid edges.
    """
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=653)
    uin = jnp.asarray(rng.uniform(-10, 10, size=(n_x, n_y + 1)))
    vin = jnp.asarray(rng.uniform(-10, 10, size=(n_x + 1, n_y)))
    dx = jnp.full((n_x, n_y + 1), 2.0)
    dy = jnp.full((n_x + 1, n_y), 2.0)
    dxa = jnp.full((n_x, n_y), 2.0)
    dya = jnp.full((n_x, n_y), 2.0)
    uout, vout = dtoa_vort_on(uin, vin, dx, dy, dxa, dya)
    # With dx=dxa=2, uout[i, j] = 0.5·(uin[i, j] + uin[i, j+1]) (simple avg)
    expected_uout = 0.5 * (uin[:, :-1] + uin[:, 1:])
    expected_vout = 0.5 * (vin[:-1, :] + vin[1:, :])
    assert jnp.allclose(uout, expected_uout, atol=1e-12)
    assert jnp.allclose(vout, expected_vout, atol=1e-12)


def test_dtoa_batched():
    """Leading batch axes preserved."""
    n_x, n_y = 4, 4
    n_lev = 3
    rng = np.random.default_rng(seed=654)
    uin = jnp.asarray(rng.uniform(-1, 1, size=(n_lev, n_x, n_y + 1)))
    vin = jnp.asarray(rng.uniform(-1, 1, size=(n_lev, n_x + 1, n_y)))
    dx = jnp.full((n_lev, n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_lev, n_x + 1, n_y), 1000.0)
    dxa = jnp.full((n_lev, n_x, n_y), 1000.0)
    dya = jnp.full((n_lev, n_x, n_y), 1000.0)
    uout, vout = dtoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert uout.shape == (n_lev, n_x, n_y)
    assert vout.shape == (n_lev, n_x, n_y)
