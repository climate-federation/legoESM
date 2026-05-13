"""FV3_3D iter 860: spi_z_score_fv3.

SPI = (P − μ_P) / σ_P  (z-score drought index).

Tests
-----

1. ``test_at_climatology_zero``: P = μ_P → SPI = 0.
2. ``test_extreme_drought_le_minus2``: P = μ−2σ → SPI = −2 (extreme).
3. ``test_extreme_wet_ge_plus2``: P = μ+2σ → SPI = +2.
4. ``test_drought_2012_us``: P=−1.5σ → moderate-to-severe drought.
5. ``test_sigma_zero_floored``: σ=0 → finite (no NaN).
6. ``test_monotone``: ↑P → ↑SPI.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import spi_z_score_fv3


def test_at_climatology_zero():
    """P = μ_P → SPI = 0."""
    spi = spi_z_score_fv3(
        precip=jnp.array([100.0]),
        mu_p=jnp.array([100.0]),
        sigma_p=jnp.array([30.0]),
    )
    np.testing.assert_allclose(np.asarray(spi), [0.0], atol=1e-12)


def test_extreme_drought_le_minus2():
    """P = μ−2σ → SPI = −2 (extreme drought threshold)."""
    spi = spi_z_score_fv3(
        precip=jnp.array([40.0]),
        mu_p=jnp.array([100.0]),
        sigma_p=jnp.array([30.0]),
    )
    np.testing.assert_allclose(np.asarray(spi), [-2.0], rtol=1e-12)


def test_extreme_wet_ge_plus2():
    """P = μ+2σ → SPI = +2."""
    spi = spi_z_score_fv3(
        precip=jnp.array([160.0]),
        mu_p=jnp.array([100.0]),
        sigma_p=jnp.array([30.0]),
    )
    np.testing.assert_allclose(np.asarray(spi), [2.0], rtol=1e-12)


def test_drought_2012_us():
    """2012 US Midwest drought peak SPI ≈ −1.5 (severe)."""
    spi = spi_z_score_fv3(
        precip=jnp.array([55.0]),
        mu_p=jnp.array([100.0]),
        sigma_p=jnp.array([30.0]),
    )
    # (55-100)/30 = -1.5
    np.testing.assert_allclose(np.asarray(spi), [-1.5], rtol=1e-12)


def test_sigma_zero_floored():
    """σ=0 → finite via floor."""
    spi = spi_z_score_fv3(
        precip=jnp.array([100.0]),
        mu_p=jnp.array([100.0]),
        sigma_p=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(spi))


def test_monotone():
    """↑P at fixed (μ, σ) → ↑SPI."""
    mu = jnp.full((4,), 100.0)
    sig = jnp.full((4,), 30.0)
    p = jnp.array([50.0, 80.0, 120.0, 160.0])
    spi = spi_z_score_fv3(p, mu, sig)
    diffs = jnp.diff(spi)
    assert jnp.all(diffs > 0.0)


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=860)
    n_x, n_y = 6, 8
    p = jnp.asarray(rng.uniform(0.0, 200.0, size=(n_x, n_y)))
    mu = jnp.asarray(rng.uniform(50.0, 150.0, size=(n_x, n_y)))
    sig = jnp.asarray(rng.uniform(10.0, 50.0, size=(n_x, n_y)))
    spi = spi_z_score_fv3(p, mu, sig)
    assert spi.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(spi))
