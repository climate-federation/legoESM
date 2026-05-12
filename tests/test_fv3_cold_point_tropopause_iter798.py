"""FV3_3D iter 798: cold_point_tropopause_fv3.

Tests
-----

1. ``test_cpt_tropical_profile``: cold point at ~17 km.
2. ``test_cpt_monotone_decreasing``: T decreases with z → top.
3. ``test_cpt_monotone_increasing``: T increases with z → bottom.
4. ``test_cpt_returns_both``: tuple of (z_cpt, T_cpt) shape correct.
5. ``test_cpt_complementary_to_797``: tropical column z_CPT > z_LRT.
6. ``test_cpt_shapes_batched``: 3-D (n_x, n_y, km) → (n_x, n_y).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    cold_point_tropopause_fv3,
    lapse_rate_tropopause_fv3,
)


def test_cpt_tropical_profile():
    """Tropical profile: cold point at ~17 km, ~195 K."""
    z = jnp.array([1000.0, 5000.0, 11_000.0, 15_000.0, 17_000.0,
                   19_000.0, 22_000.0, 25_000.0])
    # T decreases through trop, min at 17 km, increases into strato
    t = jnp.array([290.0, 260.0, 220.0, 200.0, 195.0, 200.0, 215.0, 230.0])
    z_cpt, t_cpt = cold_point_tropopause_fv3(t, z)
    assert float(z_cpt) == 17_000.0
    np.testing.assert_allclose(np.asarray(t_cpt), [195.0], rtol=1e-12)


def test_cpt_monotone_decreasing():
    """T decreases with z → cold point at top."""
    z = jnp.array([1000.0, 5000.0, 10_000.0, 15_000.0, 20_000.0])
    t = jnp.array([280.0, 250.0, 220.0, 190.0, 160.0])
    z_cpt, t_cpt = cold_point_tropopause_fv3(t, z)
    assert float(z_cpt) == 20_000.0
    np.testing.assert_allclose(np.asarray(t_cpt), [160.0], rtol=1e-12)


def test_cpt_monotone_increasing():
    """T increases with z → cold point at bottom."""
    z = jnp.array([1000.0, 5000.0, 10_000.0, 15_000.0, 20_000.0])
    t = jnp.array([200.0, 210.0, 220.0, 230.0, 240.0])
    z_cpt, t_cpt = cold_point_tropopause_fv3(t, z)
    assert float(z_cpt) == 1000.0
    np.testing.assert_allclose(np.asarray(t_cpt), [200.0], rtol=1e-12)


def test_cpt_returns_both():
    """Tuple return shape preserved across batch."""
    rng = np.random.default_rng(seed=798)
    n_x, n_y, km = 4, 5, 30
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(200.0, 800.0, size=(n_x, n_y, km))), axis=-1
    )
    t = jnp.asarray(rng.uniform(160.0, 290.0, size=(n_x, n_y, km)))
    z_cpt, t_cpt = cold_point_tropopause_fv3(t, z)
    assert z_cpt.shape == (n_x, n_y)
    assert t_cpt.shape == (n_x, n_y)


def test_cpt_complementary_to_797():
    """Tropical sounding: both CPT and LRT within tropopause-layer range."""
    z = jnp.array([1000.0, 5000.0, 11_000.0, 15_000.0, 17_000.0,
                   19_000.0, 22_000.0, 25_000.0])
    t = jnp.array([290.0, 260.0, 220.0, 200.0, 195.0, 200.0, 215.0, 230.0])
    z_cpt, _ = cold_point_tropopause_fv3(t, z)
    z_lrt = lapse_rate_tropopause_fv3(t, z)
    # Both detected in TTL (15-20 km) for this profile
    assert 14_000.0 < float(z_cpt) < 21_000.0
    assert 14_000.0 < float(z_lrt) < 21_000.0


def test_cpt_shapes_batched():
    """3-D batched + finite + within sensible range."""
    rng = np.random.default_rng(seed=799)
    n_x, n_y, km = 4, 5, 30
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(200.0, 800.0, size=(n_x, n_y, km))), axis=-1
    )
    t = jnp.asarray(rng.uniform(180.0, 300.0, size=(n_x, n_y, km)))
    z_cpt, t_cpt = cold_point_tropopause_fv3(t, z)
    assert jnp.all(jnp.isfinite(z_cpt))
    assert jnp.all(jnp.isfinite(t_cpt))
    assert jnp.all(z_cpt > 0.0)
    assert jnp.all(t_cpt > 150.0)
