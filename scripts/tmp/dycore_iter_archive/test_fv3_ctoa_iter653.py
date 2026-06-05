"""FV3_3D iter 653: ctoa_vort_on port.

Faithful JAX port of FV3 ``ctoa`` (tools/test_cases.F90:8114-8174,
simple/circulation-conserving branch).  C-grid → A-grid winds.

Tests
-----

1. ``test_ctoa_shape``.
2. ``test_ctoa_zero_winds``.
3. ``test_ctoa_uniform_winds``.
4. ``test_ctoa_simple_average_when_dx_dxa_equal``.
5. ``test_ctoa_batched``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import ctoa_vort_on


def test_ctoa_shape():
    """Output (n_x, n_y) from C-grid inputs (n_x+1, n_y), (n_x, n_y+1)."""
    n_x, n_y = 5, 7
    rng = np.random.default_rng(seed=653)
    uin = jnp.asarray(rng.uniform(-1, 1, size=(n_x + 1, n_y)))
    vin = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y + 1)))
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = ctoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert uout.shape == (n_x, n_y)
    assert vout.shape == (n_x, n_y)


def test_ctoa_zero_winds():
    """Zero C-grid winds → zero A-grid output."""
    n_x, n_y = 4, 4
    uin = jnp.zeros((n_x + 1, n_y))
    vin = jnp.zeros((n_x, n_y + 1))
    dx = jnp.full((n_x, n_y + 1), 500.0)
    dy = jnp.full((n_x + 1, n_y), 500.0)
    dxa = jnp.full((n_x, n_y), 500.0)
    dya = jnp.full((n_x, n_y), 500.0)
    uout, vout = ctoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert jnp.allclose(uout, 0.0, atol=1e-14)
    assert jnp.allclose(vout, 0.0, atol=1e-14)


def test_ctoa_uniform_winds():
    """Uniform u=U, v=V with uniform dx, dy → uout=U, vout=V."""
    n_x, n_y = 4, 4
    U, V = 7.0, 3.0
    uin = jnp.full((n_x + 1, n_y), U)
    vin = jnp.full((n_x, n_y + 1), V)
    dx = jnp.full((n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_x + 1, n_y), 1000.0)
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = ctoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert jnp.allclose(uout, U, atol=1e-12)
    assert jnp.allclose(vout, V, atol=1e-12)


def test_ctoa_simple_average_when_dx_dxa_equal():
    """With dx=dxa, dy=dya: uout = 0.5·(uin[i] + uin[i+1])."""
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=655)
    uin = jnp.asarray(rng.uniform(-10, 10, size=(n_x + 1, n_y)))
    vin = jnp.asarray(rng.uniform(-10, 10, size=(n_x, n_y + 1)))
    dx = jnp.full((n_x, n_y + 1), 2.0)
    dy = jnp.full((n_x + 1, n_y), 2.0)
    dxa = jnp.full((n_x, n_y), 2.0)
    dya = jnp.full((n_x, n_y), 2.0)
    uout, vout = ctoa_vort_on(uin, vin, dx, dy, dxa, dya)
    expected_uout = 0.5 * (uin[:-1, :] + uin[1:, :])
    expected_vout = 0.5 * (vin[:, :-1] + vin[:, 1:])
    assert jnp.allclose(uout, expected_uout, atol=1e-12)
    assert jnp.allclose(vout, expected_vout, atol=1e-12)


def test_ctoa_batched():
    """Leading batch axes preserved."""
    n_x, n_y, n_lev = 4, 4, 3
    rng = np.random.default_rng(seed=656)
    uin = jnp.asarray(rng.uniform(-1, 1, size=(n_lev, n_x + 1, n_y)))
    vin = jnp.asarray(rng.uniform(-1, 1, size=(n_lev, n_x, n_y + 1)))
    dx = jnp.full((n_lev, n_x, n_y + 1), 1000.0)
    dy = jnp.full((n_lev, n_x + 1, n_y), 1000.0)
    dxa = jnp.full((n_lev, n_x, n_y), 1000.0)
    dya = jnp.full((n_lev, n_x, n_y), 1000.0)
    uout, vout = ctoa_vort_on(uin, vin, dx, dy, dxa, dya)
    assert uout.shape == (n_lev, n_x, n_y)
    assert vout.shape == (n_lev, n_x, n_y)
