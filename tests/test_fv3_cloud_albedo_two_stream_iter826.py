"""FV3_3D iter 826: cloud_albedo_two_stream_fv3 (Coakley-Chylek 1975).

α = τ(1-g) / (2μ_0 + τ(1-g)).

Tests
-----

1. ``test_alpha_overhead_sun``: τ=15, μ_0=1, g=0.85 → α≈0.529.
2. ``test_alpha_zero_tau``: τ=0 → α=0.
3. ``test_alpha_large_tau``: τ→∞ → α→1.
4. ``test_alpha_grazing_sun``: low μ_0 → high α.
5. ``test_alpha_monotone_tau``: ↑τ → ↑α.
6. ``test_alpha_custom_g``: ice g=0.7 → higher α than water g=0.85.
7. ``test_alpha_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import cloud_albedo_two_stream_fv3


def test_alpha_overhead_sun():
    """τ=15, μ_0=1 (overhead), g=0.85 → α = 2.25/(2+2.25) = 0.529."""
    tau = jnp.array([15.0])
    mu = jnp.array([1.0])
    alpha = cloud_albedo_two_stream_fv3(tau, mu)
    expected = 15.0 * 0.15 / (2.0 + 15.0 * 0.15)
    np.testing.assert_allclose(np.asarray(alpha), [expected], rtol=1e-12)
    assert 0.5 < float(alpha[0]) < 0.6


def test_alpha_zero_tau():
    """τ=0 → α=0."""
    tau = jnp.zeros((3,))
    mu = jnp.array([0.5, 1.0, 0.1])
    alpha = cloud_albedo_two_stream_fv3(tau, mu)
    np.testing.assert_allclose(np.asarray(alpha), jnp.zeros((3,)), atol=1e-15)


def test_alpha_large_tau():
    """τ→∞ → α→1."""
    tau = jnp.array([1e6])
    mu = jnp.array([0.5])
    alpha = cloud_albedo_two_stream_fv3(tau, mu)
    assert float(alpha[0]) > 0.999


def test_alpha_grazing_sun():
    """Low μ_0 → higher α (saturating)."""
    tau = jnp.array([10.0, 10.0])
    mu_high = jnp.array([1.0])
    mu_low = jnp.array([0.1])
    alpha_high = cloud_albedo_two_stream_fv3(tau[:1], mu_high)
    alpha_low = cloud_albedo_two_stream_fv3(tau[:1], mu_low)
    assert float(alpha_low[0]) > float(alpha_high[0])


def test_alpha_monotone_tau():
    """↑τ → ↑α at fixed μ_0."""
    tau = jnp.array([1.0, 5.0, 10.0, 50.0])
    mu = jnp.full((4,), 0.7)
    alpha = cloud_albedo_two_stream_fv3(tau, mu)
    assert jnp.all(jnp.diff(alpha) > 0.0)


def test_alpha_custom_g():
    """Ice clouds g=0.7 → higher α than water g=0.85 at same τ.

    (1-g)=0.3 vs 0.15; larger factor → larger α."""
    tau = jnp.array([10.0])
    mu = jnp.array([1.0])
    alpha_water = cloud_albedo_two_stream_fv3(tau, mu, g=0.85)
    alpha_ice = cloud_albedo_two_stream_fv3(tau, mu, g=0.7)
    assert float(alpha_ice[0]) > float(alpha_water[0])


def test_alpha_shapes_finite():
    """3-D shapes preserved, finite, in [0, 1]."""
    rng = np.random.default_rng(seed=826)
    n_x, n_y = 6, 8
    tau = jnp.asarray(rng.uniform(0.0, 50.0, size=(n_x, n_y)))
    mu = jnp.asarray(rng.uniform(0.01, 1.0, size=(n_x, n_y)))
    alpha = cloud_albedo_two_stream_fv3(tau, mu)
    assert alpha.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(alpha))
    assert jnp.all(alpha >= 0.0)
    assert jnp.all(alpha <= 1.0)
