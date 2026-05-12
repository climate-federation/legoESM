"""FV3_3D iter 743: refactor iter-677 z_sum + iter-694 prt_mass to use iter-742.

Tests
-----

1. ``test_z_sum_matches_inline``.
2. ``test_prt_mass_column_matches_inline``.
3. ``test_iter677_unchanged``.
4. ``test_iter694_unchanged``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    column_integral_delp_fv3,
    prt_mass_fv3,
    z_sum_fv3,
)


def test_z_sum_matches_inline():
    """iter-677 z_sum refactor: output = Σ delp·q (no /g)."""
    rng = np.random.default_rng(seed=743)
    km = 10
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(km,)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(km,)))
    out = z_sum_fv3(delp, q)
    expected = jnp.sum(delp * q)
    assert abs(float(out) - float(expected)) < 1e-10


def test_prt_mass_column_matches_inline():
    """iter-694 col_mass = sum(delp*q)/g."""
    rng = np.random.default_rng(seed=744)
    n_x, n_y, km = 3, 3, 10
    delp = jnp.full((n_x, n_y, km), 1.0e4)
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    col = column_integral_delp_fv3(q, delp)
    expected = jnp.sum(delp * q, axis=-1) / constants.g
    assert jnp.allclose(col, expected, atol=1e-10)


def test_iter677_unchanged():
    """iter-677 z_sum still sums delp·q correctly."""
    delp = jnp.full((3, 3, 5), 1000.0)
    q = jnp.full((3, 3, 5), 0.01)
    out = z_sum_fv3(delp, q)
    expected = jnp.full((3, 3), 5.0 * 1000.0 * 0.01)
    assert jnp.allclose(out, expected, atol=1e-10)


def test_iter694_unchanged():
    """iter-694 prt_mass column water unchanged after refactor."""
    n_x, n_y, km = 4, 4, 10
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 1.0e4)
    area = jnp.ones((n_x, n_y))
    q = jnp.full((n_x, n_y, km), 0.01)
    diag = prt_mass_fv3(ps, delp, {"sphum": q}, area)
    expected = 0.01 * 1.0e5 / constants.g
    assert abs(diag["sphum"] - expected) / expected < 1e-10
