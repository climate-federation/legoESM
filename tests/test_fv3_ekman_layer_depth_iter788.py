"""FV3_3D iter 788: ekman_layer_depth_fv3 (δ_E = √(2K_v/|f|)).

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_delta_atmospheric_pbl``: K_v=10, |f|=1e-4 → δ_E ≈ 447 m.
2. ``test_delta_ocean_ml``: K_v=0.01, |f|=1e-4 → δ_E ≈ 14 m.
3. ``test_delta_zero_K``: K_v=0 → δ_E=0.
4. ``test_delta_monotonic_K``: ↑K_v → ↑δ_E; ↑|f| → ↓δ_E.
5. ``test_delta_equator_floored``: f=0 → huge but finite.
6. ``test_delta_composes_iter778``: (K_v, lat) → f → δ_E.
7. ``test_delta_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    ekman_layer_depth_fv3,
)


def test_delta_atmospheric_pbl():
    """K_v=10 m²/s, |f|=1e-4 → δ_E = √(2·10/1e-4) ≈ 447 m."""
    K = jnp.array([10.0])
    f = jnp.array([1.0e-4])
    delta = ekman_layer_depth_fv3(K, f)
    expected = jnp.sqrt(2.0 * 10.0 / 1.0e-4)
    np.testing.assert_allclose(np.asarray(delta), [expected], rtol=1e-12)
    # Sanity: ~447 m
    assert 400.0 < float(delta[0]) < 500.0


def test_delta_ocean_ml():
    """K_v=0.01 m²/s, |f|=1e-4 → δ_E = √(0.02/1e-4) ≈ 14 m."""
    K = jnp.array([0.01])
    f = jnp.array([1.0e-4])
    delta = ekman_layer_depth_fv3(K, f)
    expected = jnp.sqrt(2.0 * 0.01 / 1.0e-4)
    np.testing.assert_allclose(np.asarray(delta), [expected], rtol=1e-12)
    # Sanity: ~14.1 m
    assert 12.0 < float(delta[0]) < 16.0


def test_delta_zero_K():
    """K_v=0 → δ_E=0."""
    K = jnp.array([0.0, 0.0, 0.0])
    f = jnp.array([1e-5, 1e-4, 1.5e-4])
    delta = ekman_layer_depth_fv3(K, f)
    np.testing.assert_allclose(np.asarray(delta), [0.0, 0.0, 0.0], atol=1e-15)


def test_delta_monotonic_K():
    """↑K_v → ↑δ_E; ↑|f| → ↓δ_E."""
    f_base = jnp.array([1e-4, 1e-4, 1e-4])
    K_lo = jnp.array([1.0, 5.0, 10.0])
    K_hi = K_lo + 5.0
    assert jnp.all(ekman_layer_depth_fv3(K_hi, f_base) > ekman_layer_depth_fv3(K_lo, f_base))

    K_base = jnp.array([10.0, 10.0, 10.0])
    f_lo = jnp.array([0.5e-4, 1.0e-4, 1.5e-4])
    f_hi = f_lo + 0.5e-4
    assert jnp.all(ekman_layer_depth_fv3(K_base, f_hi) < ekman_layer_depth_fv3(K_base, f_lo))


def test_delta_equator_floored():
    """f=0 → δ_E huge but finite via f_floor."""
    K = jnp.array([10.0])
    f = jnp.array([0.0])
    delta = ekman_layer_depth_fv3(K, f, f_floor=1e-12)
    assert jnp.all(jnp.isfinite(delta))
    # sqrt(2·10/1e-12) ≈ 4.5e6 m
    assert float(delta[0]) > 1e6


def test_delta_composes_iter778():
    """Pipeline (K_v, lat) → f → δ_E."""
    K = jnp.array([10.0])
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    delta = ekman_layer_depth_fv3(K, f)
    # |f|=2·Ω·sin(45°) ≈ 1.03e-4 → δ_E ≈ √(2·10/1.03e-4) ≈ 441 m
    assert 400.0 < float(delta[0]) < 500.0


def test_delta_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=788)
    n_x, n_y, km = 4, 5, 20
    K = jnp.asarray(rng.uniform(0.0, 20.0, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(1e-5, 1.5e-4, size=(n_x, n_y, km)))
    delta = ekman_layer_depth_fv3(K, f)
    assert delta.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(delta >= 0.0)
