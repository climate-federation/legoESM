"""FV3_3D iter 725: omega_diagnostic_fv3 port.

Pressure vertical velocity ω = dp/dt in the hydrostatic limit
(ω = w · delp/delz = -ρ·g·w).  FV3 dyn_core.F90:1642 uses full
Lagrangian form; this diagnostic captures the dominant
w·∂p/∂z term.

Tests
-----

1. ``test_omega_zero_w``.
2. ``test_omega_descending_air_positive``.
3. ``test_omega_known_density``.
4. ``test_omega_shapes_3d``.
5. ``test_omega_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import omega_diagnostic_fv3


def test_omega_zero_w():
    """w = 0 → ω = 0."""
    km = 10
    w = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -500.0)
    omega = omega_diagnostic_fv3(w, delp, delz)
    assert jnp.all(jnp.abs(omega) < 1e-15)


def test_omega_descending_air_positive():
    """w < 0 (descending) → ω > 0 (pressure increasing)."""
    km = 5
    w = jnp.full((km,), -0.1)   # 10 cm/s descent
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    omega = omega_diagnostic_fv3(w, delp, delz)
    assert jnp.all(omega > 0.0)


def test_omega_known_density():
    """ρ = -delp/(g·delz), ω = -ρ·g·w → check against analytic.

    delp = 1e4 Pa, delz = -1000 m → ρ = 10000/(9.81·1000) = 1.0193 kg/m³
    w = 0.01 m/s → ω = -1.0193·9.81·0.01 = -0.10 Pa/s
    """
    km = 5
    w = jnp.full((km,), 0.01)
    delp = jnp.full((km,), 1.0e4)
    delz = jnp.full((km,), -1000.0)
    omega = omega_diagnostic_fv3(w, delp, delz)
    rho = -1.0e4 / (constants.g * -1000.0)   # = 1.0e4/(g·1000) ~ 1.0193
    expected = -rho * constants.g * 0.01
    assert jnp.all(jnp.abs(omega - expected) < 1e-10)


def test_omega_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=725)
    n_x, n_y, km = 4, 5, 20
    w = jnp.asarray(rng.normal(scale=0.5, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    delz = jnp.full((n_x, n_y, km), -200.0)
    omega = omega_diagnostic_fv3(w, delp, delz)
    assert omega.shape == (n_x, n_y, km)


def test_omega_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=726)
    km = 30
    w = jnp.asarray(rng.normal(scale=1.0, size=(4, 4, km)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, km)))
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(4, 4, km)))
    omega = omega_diagnostic_fv3(w, delp, delz)
    assert jnp.all(jnp.isfinite(omega))
