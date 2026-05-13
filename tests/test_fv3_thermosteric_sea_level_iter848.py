"""FV3_3D iter 848: thermosteric_sea_level_fv3.

Δη = Σ_k α_T·ΔT_k·H_k  [m].

Tests
-----

1. ``test_uniform_warming_4km``: ΔT=0.1 K over 4000 m → Δη ≈ 8 cm.
2. ``test_no_warming_zero``: ΔT=0 → Δη=0.
3. ``test_cooling_negative``: ΔT<0 → Δη<0.
4. ``test_pinatubo_transient``: −0.05 K upper 1000 m → Δη ≈ −1 cm.
5. ``test_lgm_lowering``: −3 K over 5000 m → Δη ≈ −3 m.
6. ``test_alpha_t_scaling``: 2× α_T → 2× Δη.
7. ``test_batched_columns``: 3-D (lat×lon×depth) shapes preserved.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import thermosteric_sea_level_fv3


def test_uniform_warming_4km():
    """ΔT=0.1 K uniform over 4000 m → Δη ≈ 8 cm."""
    n_layers = 40
    dt = jnp.full((n_layers,), 0.1)
    h = jnp.full((n_layers,), 100.0)  # 100 m/layer × 40 = 4000 m
    eta = thermosteric_sea_level_fv3(dt, h)
    np.testing.assert_allclose(np.asarray(eta), 0.08, rtol=1e-12)


def test_no_warming_zero():
    """ΔT=0 → Δη=0."""
    dt = jnp.zeros(20)
    h = jnp.full((20,), 200.0)
    eta = thermosteric_sea_level_fv3(dt, h)
    np.testing.assert_allclose(np.asarray(eta), 0.0, atol=1e-14)


def test_cooling_negative():
    """ΔT<0 → Δη<0 (thermal contraction)."""
    dt = jnp.full((10,), -0.05)
    h = jnp.full((10,), 100.0)
    eta = thermosteric_sea_level_fv3(dt, h)
    assert float(eta) < 0.0


def test_pinatubo_transient():
    """−0.05 K over upper 1000 m → Δη ≈ −1 cm."""
    dt = jnp.full((10,), -0.05)
    h = jnp.full((10,), 100.0)
    eta = thermosteric_sea_level_fv3(dt, h)
    np.testing.assert_allclose(np.asarray(eta), -0.01, rtol=1e-12)


def test_lgm_lowering():
    """−3 K over full-depth 5000 m → Δη ≈ −3 m (LGM thermal contribution)."""
    dt = jnp.full((50,), -3.0)
    h = jnp.full((50,), 100.0)
    eta = thermosteric_sea_level_fv3(dt, h)
    np.testing.assert_allclose(np.asarray(eta), -3.0, rtol=1e-12)


def test_alpha_t_scaling():
    """2× α_T → 2× Δη (linear in expansion coefficient)."""
    dt = jnp.full((10,), 0.1)
    h = jnp.full((10,), 100.0)
    eta1 = thermosteric_sea_level_fv3(dt, h, alpha_t=2.0e-4)
    eta2 = thermosteric_sea_level_fv3(dt, h, alpha_t=4.0e-4)
    np.testing.assert_allclose(np.asarray(eta2), 2.0 * np.asarray(eta1),
                                rtol=1e-12)


def test_batched_columns():
    """3-D (lat, lon, depth) → 2-D (lat, lon) sum-over-depth."""
    rng = np.random.default_rng(seed=848)
    n_lat, n_lon, n_z = 4, 5, 30
    dt = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_lat, n_lon, n_z)))
    h = jnp.asarray(rng.uniform(50.0, 200.0, size=(n_lat, n_lon, n_z)))
    eta = thermosteric_sea_level_fv3(dt, h)
    assert eta.shape == (n_lat, n_lon)


def test_shapes_finite():
    """Random batched arrays preserve shape, finite."""
    rng = np.random.default_rng(seed=848)
    n_x, n_y, n_z = 6, 8, 25
    dt = jnp.asarray(rng.uniform(-2.0, 2.0, size=(n_x, n_y, n_z)))
    h = jnp.asarray(rng.uniform(20.0, 300.0, size=(n_x, n_y, n_z)))
    alpha = jnp.asarray(rng.uniform(1e-5, 4e-4, size=(n_x, n_y, n_z)))
    eta = thermosteric_sea_level_fv3(dt, h, alpha)
    assert eta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(eta))
