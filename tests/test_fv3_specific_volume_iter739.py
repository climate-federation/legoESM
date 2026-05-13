"""FV3_3D iter 739: specific_volume_fv3 port.

alpha = 1/rho = -g*delz/delp.

Tests
-----

1. ``test_alpha_inverse_of_rho``.
2. ``test_alpha_known_value``.
3. ``test_alpha_positive``.
4. ``test_alpha_shapes_3d``.
5. ``test_alpha_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import air_density_fv3, specific_volume_fv3


def test_alpha_inverse_of_rho():
    """alpha * rho = 1 exactly."""
    rng = np.random.default_rng(seed=739)
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(10,)))
    delz = jnp.asarray(rng.uniform(-500.0, -50.0, size=(10,)))
    alpha = specific_volume_fv3(delp, delz)
    rho = air_density_fv3(delp, delz)
    assert jnp.allclose(alpha * rho, 1.0, atol=1e-12)


def test_alpha_known_value():
    """delp=1e4, delz=-1000 → alpha = g·1000/1e4 = 9.81e-1 m³/kg."""
    delp = jnp.array([1.0e4])
    delz = jnp.array([-1000.0])
    alpha = specific_volume_fv3(delp, delz)
    expected = constants.g * 1000.0 / 1.0e4
    assert abs(float(alpha[0]) - expected) < 1e-10


def test_alpha_positive():
    """FV3 delp>0, delz<0 → alpha > 0."""
    rng = np.random.default_rng(seed=740)
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(20,)))
    delz = jnp.asarray(rng.uniform(-500.0, -50.0, size=(20,)))
    alpha = specific_volume_fv3(delp, delz)
    assert jnp.all(alpha > 0.0)


def test_alpha_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=741)
    n_x, n_y, km = 4, 5, 20
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(n_x, n_y, km)))
    alpha = specific_volume_fv3(delp, delz)
    assert alpha.shape == (n_x, n_y, km)


def test_alpha_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=742)
    km = 30
    delp = jnp.asarray(rng.uniform(100.0, 3000.0, size=(4, 4, km)))
    delz = jnp.asarray(rng.uniform(-1000.0, -50.0, size=(4, 4, km)))
    alpha = specific_volume_fv3(delp, delz)
    assert jnp.all(jnp.isfinite(alpha))
