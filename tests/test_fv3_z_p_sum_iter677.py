"""FV3_3D iter 677: z_sum_fv3 + p_sum_fv3 ports.

Faithful JAX ports of FV3 ``z_sum`` (tools/fv_diagnostics.F90:
4265-4285) + ``p_sum`` (4287-4310, serial branch).

Tests
-----

1. ``test_z_sum_constant_field``.
2. ``test_z_sum_linear_field``.
3. ``test_z_sum_batched``.
4. ``test_p_sum_uniform_delp``.
5. ``test_p_sum_area_weighted``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import p_sum_fv3, z_sum_fv3


def test_z_sum_constant_field():
    """q=const, delp=1 → z_sum = const · km."""
    km = 10
    delp = jnp.ones((4, 4, km))
    q = jnp.full((4, 4, km), 3.5)
    out = z_sum_fv3(delp, q)
    assert jnp.allclose(out, 3.5 * km, atol=1e-12)


def test_z_sum_linear_field():
    """delp·q sums column-wise."""
    rng = np.random.default_rng(seed=677)
    delp = jnp.asarray(rng.uniform(500, 3000, size=(4, 5)))
    q = jnp.asarray(rng.uniform(-1, 1, size=(4, 5)))
    out = z_sum_fv3(delp, q)
    expected = jnp.sum(delp * q, axis=-1)
    assert jnp.allclose(out, expected, atol=1e-12)


def test_z_sum_batched():
    """Leading axes preserved."""
    delp = jnp.ones((6, 8, 8, 10))
    q = jnp.full((6, 8, 8, 10), 2.0)
    out = z_sum_fv3(delp, q)
    assert out.shape == (6, 8, 8)


def test_p_sum_uniform_delp():
    """All cells delp=p_total/km, area=A → p_sum = p_total."""
    km = 10
    p_total = 1.0e5
    delp = jnp.full((4, 4, km), p_total / km)
    area = jnp.full((4, 4), 1.0e10)
    out = float(p_sum_fv3(delp, area))
    assert abs(out - p_total) < 1e-6


def test_p_sum_area_weighted():
    """Variable col_sum but uniform area → simple mean."""
    rng = np.random.default_rng(seed=678)
    km = 5
    col_sum = jnp.asarray(rng.uniform(8e4, 1.1e5, size=(6, 6)))
    # delp such that sum(delp) along last axis = col_sum:
    delp = jnp.broadcast_to(col_sum[..., None] / km, col_sum.shape + (km,))
    area = jnp.ones((6, 6))
    out = float(p_sum_fv3(delp, area))
    expected = float(jnp.mean(col_sum))
    assert abs(out - expected) / expected < 1e-12
