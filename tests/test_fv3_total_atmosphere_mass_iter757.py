"""FV3_3D iter 757: total_atmosphere_mass_fv3 port.

M_col = sum_k delp / g  (kg/m^2).

Tests
-----

1. ``test_mass_known_earth_value``.
2. ``test_mass_zero_delp``.
3. ``test_mass_matches_ps_minus_ptop_over_g``.
4. ``test_mass_dry_plus_water_decomposition``.
5. ``test_mass_shapes_3d``.
6. ``test_mass_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dry_surface_pressure_fv3,
    total_atmosphere_mass_fv3,
    total_water_column_fv3,
)


def test_mass_known_earth_value():
    """p_s=1e5, p_top=0 → M_col ≈ 10197 kg/m²."""
    km = 10
    delp = jnp.full((km,), 1.0e4)   # sum = 1e5 Pa
    m_col = total_atmosphere_mass_fv3(delp)
    expected = 1.0e5 / constants.g
    assert abs(float(m_col) - expected) / expected < 1e-10


def test_mass_zero_delp():
    """Zero delp → 0 mass."""
    km = 5
    delp = jnp.zeros((km,))
    m_col = total_atmosphere_mass_fv3(delp)
    assert abs(float(m_col)) < 1e-15


def test_mass_matches_ps_minus_ptop_over_g():
    """M_col = Σ delp / g = (p_s − p_top) / g."""
    rng = np.random.default_rng(seed=757)
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(20,)))
    m_col = total_atmosphere_mass_fv3(delp)
    expected = float(jnp.sum(delp)) / constants.g
    assert abs(float(m_col) - expected) < 1e-8


def test_mass_dry_plus_water_decomposition():
    """M_col = M_dry + TWC.

    Verify: M_col − TWC = (ps − g·TWC)/g = ps_dry/g (per cell)."""
    n_x, n_y, km = 3, 3, 10
    ps = jnp.full((n_x, n_y), 1.0e5)
    delp = jnp.full((n_x, n_y, km), 1.0e4)
    q = jnp.full((n_x, n_y, km), 0.01)
    m_col = total_atmosphere_mass_fv3(delp)
    twc = total_water_column_fv3(delp, q_sphum=q)
    ps_dry = dry_surface_pressure_fv3(ps, delp, q_sphum=q)
    # ps_dry / g = M_col - TWC
    expected = ps_dry / constants.g
    assert jnp.allclose(m_col - twc, expected, atol=1e-6)


def test_mass_shapes_3d():
    """3-D shape → 2-D output."""
    rng = np.random.default_rng(seed=758)
    n_x, n_y, km = 4, 5, 20
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    m_col = total_atmosphere_mass_fv3(delp)
    assert m_col.shape == (n_x, n_y)


def test_mass_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=759)
    delp = jnp.asarray(rng.uniform(100.0, 3000.0, size=(4, 4, 30)))
    m_col = total_atmosphere_mass_fv3(delp)
    assert jnp.all(jnp.isfinite(m_col))
