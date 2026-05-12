"""FV3_3D iter 754: dry_static_energy_fv3 port.

DSE = c_p * T + g * z.

Tests
-----

1. ``test_dse_surface_matches_cpT``.
2. ``test_dse_aloft_higher``.
3. ``test_dse_known_value``.
4. ``test_dse_mse_decomposition``.
5. ``test_dse_custom_cp``.
6. ``test_dse_shapes_3d``.
7. ``test_dse_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dry_static_energy_fv3,
    moist_static_energy_fv3,
)


def test_dse_surface_matches_cpT():
    """z=0 → DSE = c_p · T."""
    T = jnp.full((5,), 290.0)
    z = jnp.zeros((5,))
    dse = dry_static_energy_fv3(T, z)
    expected = constants.c_pd * 290.0
    assert jnp.allclose(dse, expected, atol=1e-8)


def test_dse_aloft_higher():
    """z aloft → DSE > c_p·T (potential-energy adds)."""
    T = jnp.full((5,), 270.0)
    z = jnp.full((5,), 5000.0)
    dse = dry_static_energy_fv3(T, z)
    surface_value = constants.c_pd * 270.0
    assert jnp.all(dse > surface_value)


def test_dse_known_value():
    """T=290, z=1000 → DSE = c_pd·290 + g·1000."""
    T = jnp.array([290.0])
    z = jnp.array([1000.0])
    dse = dry_static_energy_fv3(T, z)
    expected = constants.c_pd * 290.0 + constants.g * 1000.0
    assert abs(float(dse[0]) - expected) / expected < 1e-12


def test_dse_mse_decomposition():
    """MSE = DSE + L · q exactly."""
    T = jnp.array([300.0])
    z = jnp.array([2000.0])
    q = jnp.array([0.015])
    dse = dry_static_energy_fv3(T, z)
    mse = moist_static_energy_fv3(T, z, q)
    expected_diff = constants.L_v * 0.015
    assert abs(float(mse[0] - dse[0]) - expected_diff) < 1e-6


def test_dse_custom_cp():
    """Custom c_p override."""
    T = jnp.array([280.0])
    z = jnp.array([500.0])
    cp_custom = 1100.0
    dse = dry_static_energy_fv3(T, z, cp=cp_custom)
    expected = 1100.0 * 280.0 + constants.g * 500.0
    assert abs(float(dse[0]) - expected) / expected < 1e-12


def test_dse_shapes_3d():
    """3-D shapes."""
    rng = np.random.default_rng(seed=754)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    z = jnp.asarray(rng.uniform(0.0, 20000.0, size=(n_x, n_y, km)))
    dse = dry_static_energy_fv3(T, z)
    assert dse.shape == (n_x, n_y, km)


def test_dse_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=755)
    T = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 30)))
    z = jnp.asarray(rng.uniform(0.0, 30000.0, size=(4, 30)))
    dse = dry_static_energy_fv3(T, z)
    assert jnp.all(jnp.isfinite(dse))
