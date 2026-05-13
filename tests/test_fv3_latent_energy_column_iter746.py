"""FV3_3D iter 746: latent_energy_column_fv3 port.

LE_col = L_v * sum(delp * q_sphum) / g.

Tests
-----

1. ``test_le_zero_q``.
2. ``test_le_known_value``.
3. ``test_le_custom_L``.
4. ``test_le_shapes_3d``.
5. ``test_le_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import latent_energy_column_fv3


def test_le_zero_q():
    """q=0 → LE_col=0."""
    km = 5
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    le = latent_energy_column_fv3(q, delp)
    assert abs(float(le)) < 1e-15


def test_le_known_value():
    """Uniform q=0.01, p_s=1e5 → LE = L_v · 0.01·1e5/g ≈ 2.55e8 J/m²."""
    km = 10
    q = jnp.full((km,), 0.01)
    delp = jnp.full((km,), 1.0e4)
    le = latent_energy_column_fv3(q, delp)
    expected = constants.L_v * 0.01 * 1.0e5 / constants.g
    assert abs(float(le) - expected) / expected < 1e-10


def test_le_custom_L():
    """L override (e.g. L_s for sublimation)."""
    km = 5
    q = jnp.full((km,), 0.005)
    delp = jnp.full((km,), 2.0e4)
    le_vap = latent_energy_column_fv3(q, delp)
    le_sub = latent_energy_column_fv3(q, delp, L=constants.L_s)
    # L_s > L_v → le_sub > le_vap
    assert float(le_sub) > float(le_vap)
    ratio = float(le_sub) / float(le_vap)
    expected_ratio = constants.L_s / constants.L_v
    assert abs(ratio - expected_ratio) < 1e-12


def test_le_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=746)
    n_x, n_y, km = 4, 5, 20
    q = jnp.asarray(rng.uniform(0.0, 0.02, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    le = latent_energy_column_fv3(q, delp)
    assert le.shape == (n_x, n_y)


def test_le_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=747)
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    le = latent_energy_column_fv3(q, delp)
    assert jnp.all(jnp.isfinite(le))
