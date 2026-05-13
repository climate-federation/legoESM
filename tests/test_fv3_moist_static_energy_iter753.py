"""FV3_3D iter 753: moist_static_energy_fv3 port.

MSE = c_p * T + g * z + L * q_sphum.

Tests
-----

1. ``test_mse_dry_surface_matches_cpT``.
2. ``test_mse_known_components``.
3. ``test_mse_custom_cp_moist``.
4. ``test_mse_custom_L``.
5. ``test_mse_shapes_3d``.
6. ``test_mse_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import moist_cp_fv3, moist_static_energy_fv3


def test_mse_dry_surface_matches_cpT():
    """q=0, z=0 → MSE = c_p · T."""
    T = jnp.full((5,), 290.0)
    z = jnp.zeros((5,))
    q = jnp.zeros((5,))
    mse = moist_static_energy_fv3(T, z, q)
    expected = constants.c_pd * 290.0
    assert jnp.allclose(mse, expected, atol=1e-8)


def test_mse_known_components():
    """T=300, z=1000, q=0.01 → MSE = c_pd·300 + g·1000 + L_v·0.01."""
    T = jnp.array([300.0])
    z = jnp.array([1000.0])
    q = jnp.array([0.01])
    mse = moist_static_energy_fv3(T, z, q)
    expected = constants.c_pd * 300.0 + constants.g * 1000.0 + constants.L_v * 0.01
    assert abs(float(mse[0]) - expected) / expected < 1e-12


def test_mse_custom_cp_moist():
    """Custom moist c_p from iter-714 moist_cp_fv3."""
    T = jnp.array([290.0])
    z = jnp.array([500.0])
    q = jnp.array([0.015])
    cpm, _ = moist_cp_fv3(q_sphum=q)
    mse = moist_static_energy_fv3(T, z, q, cp=cpm)
    expected = float(cpm[0]) * 290.0 + constants.g * 500.0 + constants.L_v * 0.015
    assert abs(float(mse[0]) - expected) / expected < 1e-12


def test_mse_custom_L():
    """Custom L (e.g. L_s for ice MSE)."""
    T = jnp.array([260.0])
    z = jnp.array([2000.0])
    q = jnp.array([0.001])
    mse_vap = moist_static_energy_fv3(T, z, q)
    mse_sub = moist_static_energy_fv3(T, z, q, L=constants.L_s)
    # L_s > L_v → mse_sub > mse_vap by (L_s - L_v) · q
    delta = float((mse_sub - mse_vap)[0])
    expected_delta = (constants.L_s - constants.L_v) * 0.001
    assert abs(delta - expected_delta) / expected_delta < 1e-12


def test_mse_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=753)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    z = jnp.asarray(rng.uniform(0.0, 20000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    mse = moist_static_energy_fv3(T, z, q)
    assert mse.shape == (n_x, n_y, km)


def test_mse_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=754)
    T = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 30)))
    z = jnp.asarray(rng.uniform(0.0, 30000.0, size=(4, 30)))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 30)))
    mse = moist_static_energy_fv3(T, z, q)
    assert jnp.all(jnp.isfinite(mse))
