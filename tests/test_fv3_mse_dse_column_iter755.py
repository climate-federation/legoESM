"""FV3_3D iter 755: mse_column_fv3 + dse_column_fv3 ports.

Tests
-----

1. ``test_mse_col_zero_inputs``.
2. ``test_dse_col_known_value``.
3. ``test_mse_col_minus_dse_col_eq_LE``.
4. ``test_mse_col_shapes_3d``.
5. ``test_dse_col_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dse_column_fv3,
    latent_energy_column_fv3,
    mse_column_fv3,
)


def test_mse_col_zero_inputs():
    """T=0, z=0, q=0 → MSE_col = 0."""
    km = 5
    z = jnp.zeros((km,))
    pt = jnp.zeros((km,))
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    mse_col = mse_column_fv3(pt, z, q, delp)
    assert abs(float(mse_col)) < 1e-12


def test_dse_col_known_value():
    """Isothermal T=280, z=0, p_s=1e5 → DSE_col = c_pd·280·1e5/g."""
    km = 10
    pt = jnp.full((km,), 280.0)
    z = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    dse_col = dse_column_fv3(pt, z, delp)
    expected = constants.c_pd * 280.0 * 1.0e5 / constants.g
    assert abs(float(dse_col) - expected) / expected < 1e-10


def test_mse_col_minus_dse_col_eq_LE():
    """MSE_col − DSE_col = LE_col exactly."""
    rng = np.random.default_rng(seed=755)
    km = 10
    pt = jnp.asarray(rng.uniform(250.0, 310.0, size=(km,)))
    z = jnp.asarray(rng.uniform(0.0, 10000.0, size=(km,)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(km,)))
    delp = jnp.full((km,), 1000.0)
    mse_col = mse_column_fv3(pt, z, q, delp)
    dse_col = dse_column_fv3(pt, z, delp)
    le_col = latent_energy_column_fv3(q, delp)
    assert abs(float(mse_col - dse_col) - float(le_col)) < 1e-6


def test_mse_col_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=756)
    n_x, n_y, km = 4, 5, 20
    pt = jnp.asarray(rng.uniform(240.0, 300.0, size=(n_x, n_y, km)))
    z = jnp.asarray(rng.uniform(0.0, 15000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    mse_col = mse_column_fv3(pt, z, q, delp)
    assert mse_col.shape == (n_x, n_y)


def test_dse_col_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=757)
    pt = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 4, 30)))
    z = jnp.asarray(rng.uniform(0.0, 30000.0, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    dse_col = dse_column_fv3(pt, z, delp)
    assert jnp.all(jnp.isfinite(dse_col))
