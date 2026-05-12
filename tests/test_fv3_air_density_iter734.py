"""FV3_3D iter 734: air_density_fv3 port + iter-725 refactor.

Tests
-----

1. ``test_rho_known_value``.
2. ``test_rho_positive``.
3. ``test_rho_iter725_omega_unchanged``.
4. ``test_rho_shapes_3d``.
5. ``test_rho_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import air_density_fv3, omega_diagnostic_fv3


def test_rho_known_value():
    """delp=1e4 Pa, delz=-1000 m → ρ = 1e4/(g·1000) = 1.0193 kg/m³."""
    delp = jnp.array([1.0e4])
    delz = jnp.array([-1000.0])
    rho = air_density_fv3(delp, delz)
    expected = 1.0e4 / (constants.g * 1000.0)
    assert abs(float(rho[0]) - expected) < 1e-10


def test_rho_positive():
    """FV3 delp>0, delz<0 → ρ > 0."""
    rng = np.random.default_rng(seed=734)
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(20,)))
    delz = jnp.asarray(rng.uniform(-500.0, -50.0, size=(20,)))
    rho = air_density_fv3(delp, delz)
    assert jnp.all(rho > 0.0)


def test_rho_iter725_omega_unchanged():
    """iter-725 omega refactor preserves output bit-identically.

    ω = w·delp/delz === -ρ·g·w (algebraically same)."""
    rng = np.random.default_rng(seed=735)
    km = 10
    w = jnp.asarray(rng.normal(scale=1.0, size=(km,)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(km,)))
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(km,)))
    omega = omega_diagnostic_fv3(w, delp, delz)
    # Manual: ω = w·delp/delz (original formula)
    expected = w * delp / delz
    assert jnp.allclose(omega, expected, atol=1e-12)


def test_rho_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=736)
    n_x, n_y, km = 4, 5, 20
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(n_x, n_y, km)))
    rho = air_density_fv3(delp, delz)
    assert rho.shape == (n_x, n_y, km)


def test_rho_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=737)
    km = 30
    delp = jnp.asarray(rng.uniform(100.0, 3000.0, size=(4, 4, km)))
    delz = jnp.asarray(rng.uniform(-1000.0, -50.0, size=(4, 4, km)))
    rho = air_density_fv3(delp, delz)
    assert jnp.all(jnp.isfinite(rho))
