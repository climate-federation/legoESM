"""FV3_3D iter 655: atod_vort_on port.

Circulation-conserving JAX port consistent with iter-652
dtoa_vort_on (FV3 atod analog without interpOrder dispatch).

Tests
-----

1. ``test_atod_shape``.
2. ``test_atod_zero_winds``.
3. ``test_atod_uniform_winds_interior``.
4. ``test_atod_boundary_zero``.
5. ``test_atod_simple_average_when_dya_uniform``.
6. ``test_atod_roundtrip_with_dtoa``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import atod_vort_on, dtoa_vort_on


def test_atod_shape():
    """Output uout (n_x, n_y+1); vout (n_x+1, n_y)."""
    n_x, n_y = 5, 7
    rng = np.random.default_rng(seed=655)
    uin = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y)))
    vin = jnp.asarray(rng.uniform(-1, 1, size=(n_x, n_y)))
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atod_vort_on(uin, vin, dxa, dya)
    assert uout.shape == (n_x, n_y + 1)
    assert vout.shape == (n_x + 1, n_y)


def test_atod_zero_winds():
    """Zero A-grid winds → zero D-grid output."""
    n_x, n_y = 4, 4
    uin = jnp.zeros((n_x, n_y))
    vin = jnp.zeros((n_x, n_y))
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atod_vort_on(uin, vin, dxa, dya)
    assert jnp.allclose(uout, 0.0, atol=1e-14)
    assert jnp.allclose(vout, 0.0, atol=1e-14)


def test_atod_uniform_winds_interior():
    """Uniform u=U, v=V → interior D-grid = U, V."""
    n_x, n_y = 4, 4
    U, V = 8.0, 4.0
    uin = jnp.full((n_x, n_y), U)
    vin = jnp.full((n_x, n_y), V)
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atod_vort_on(uin, vin, dxa, dya)
    assert jnp.allclose(uout[:, 1:n_y], U, atol=1e-12)
    assert jnp.allclose(vout[1:n_x, :], V, atol=1e-12)


def test_atod_boundary_zero():
    """Boundary D-grid edges = 0 (FV3 halo-fill later)."""
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=656)
    uin = jnp.asarray(rng.uniform(1, 10, size=(n_x, n_y)))
    vin = jnp.asarray(rng.uniform(1, 10, size=(n_x, n_y)))
    dxa = jnp.full((n_x, n_y), 1000.0)
    dya = jnp.full((n_x, n_y), 1000.0)
    uout, vout = atod_vort_on(uin, vin, dxa, dya)
    assert jnp.allclose(uout[:, 0], 0.0, atol=1e-14)
    assert jnp.allclose(uout[:, -1], 0.0, atol=1e-14)
    assert jnp.allclose(vout[0, :], 0.0, atol=1e-14)
    assert jnp.allclose(vout[-1, :], 0.0, atol=1e-14)


def test_atod_simple_average_when_dya_uniform():
    """With dya=const, uout[i, j] = 0.5·(uin[i, j-1] + uin[i, j])."""
    n_x, n_y = 4, 4
    rng = np.random.default_rng(seed=657)
    uin = jnp.asarray(rng.uniform(-10, 10, size=(n_x, n_y)))
    vin = jnp.asarray(rng.uniform(-10, 10, size=(n_x, n_y)))
    dxa = jnp.full((n_x, n_y), 1.5)
    dya = jnp.full((n_x, n_y), 1.5)
    uout, vout = atod_vort_on(uin, vin, dxa, dya)
    expected_uout = 0.5 * (uin[:, :-1] + uin[:, 1:])
    expected_vout = 0.5 * (vin[:-1, :] + vin[1:, :])
    assert jnp.allclose(uout[:, 1:n_y], expected_uout, atol=1e-12)
    assert jnp.allclose(vout[1:n_x, :], expected_vout, atol=1e-12)


def test_atod_roundtrip_with_dtoa():
    """Linear field: A→D→A round-trip recovers interior of A-grid."""
    n_x, n_y = 6, 6
    # Linear-in-i and linear-in-j field
    i_idx = jnp.arange(n_x, dtype=jnp.float64)
    j_idx = jnp.arange(n_y, dtype=jnp.float64)
    uin = jnp.broadcast_to(0.5 + 0.3 * j_idx, (n_x, n_y))
    vin = jnp.broadcast_to((0.7 + 0.2 * i_idx)[:, None], (n_x, n_y))
    dxa = jnp.full((n_x, n_y), 1.0)
    dya = jnp.full((n_x, n_y), 1.0)
    dx = jnp.full((n_x, n_y + 1), 1.0)
    dy = jnp.full((n_x + 1, n_y), 1.0)
    # A→D
    ud, vd = atod_vort_on(uin, vin, dxa, dya)
    # D→A
    ua, va = dtoa_vort_on(ud, vd, dx, dy, dxa, dya)
    # Linear function is preserved exactly by the linear averaging
    # in the interior cells (those with full j-1..j+1 and i-1..i+1 support).
    assert jnp.allclose(ua[:, 2:n_y - 2], uin[:, 2:n_y - 2], atol=1e-12)
    assert jnp.allclose(va[2:n_x - 2, :], vin[2:n_x - 2, :], atol=1e-12)
