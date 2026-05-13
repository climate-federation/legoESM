"""FV3_3D iter 745: kinetic_energy_column_fv3 port.

KE_col = sum_k delp * 0.5 * (ua^2 + va^2 [+ w^2]) / g.

Tests
-----

1. ``test_ke_col_zero_wind``.
2. ``test_ke_col_known_value``.
3. ``test_ke_col_w_optional``.
4. ``test_ke_col_shapes_3d``.
5. ``test_ke_col_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import kinetic_energy_column_fv3


def test_ke_col_zero_wind():
    """Zero wind → KE_col = 0."""
    km = 10
    z = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    ke_col = kinetic_energy_column_fv3(z, z, delp, w=z)
    assert abs(float(ke_col)) < 1e-15


def test_ke_col_known_value():
    """Uniform ua=10, va=0, p_s=1e5 → KE_col = 0.5·100·1e5/g = 509.68 J/m²."""
    km = 10
    ua = jnp.full((km,), 10.0)
    va = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)  # p_s = 1e5
    ke_col = kinetic_energy_column_fv3(ua, va, delp)
    expected = 0.5 * 100.0 * 1.0e5 / constants.g
    assert abs(float(ke_col) - expected) / expected < 1e-10


def test_ke_col_w_optional():
    """w=None → 2-component KE_col; w=array → 3-component."""
    km = 5
    ua = jnp.full((km,), 3.0)
    va = jnp.full((km,), 4.0)
    delp = jnp.full((km,), 2.0e4)
    w_arr = jnp.full((km,), 12.0)
    ke_2d = kinetic_energy_column_fv3(ua, va, delp)
    ke_3d = kinetic_energy_column_fv3(ua, va, delp, w=w_arr)
    p_s = 1.0e5
    expected_2d = 0.5 * (9.0 + 16.0) * p_s / constants.g
    expected_3d = 0.5 * (9.0 + 16.0 + 144.0) * p_s / constants.g
    assert abs(float(ke_2d) - expected_2d) / expected_2d < 1e-10
    assert abs(float(ke_3d) - expected_3d) / expected_3d < 1e-10


def test_ke_col_shapes_3d():
    """3-D (n_x, n_y, km) → 2-D (n_x, n_y) output."""
    rng = np.random.default_rng(seed=745)
    n_x, n_y, km = 4, 5, 20
    ua = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    ke_col = kinetic_energy_column_fv3(ua, va, delp)
    assert ke_col.shape == (n_x, n_y)


def test_ke_col_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=746)
    km = 30
    ua = jnp.asarray(rng.normal(scale=20.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=20.0, size=(4, 4, km)))
    w = jnp.asarray(rng.normal(scale=1.0, size=(4, 4, km)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, km)))
    ke_col = kinetic_energy_column_fv3(ua, va, delp, w=w)
    assert jnp.all(jnp.isfinite(ke_col))
