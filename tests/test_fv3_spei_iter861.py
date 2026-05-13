"""FV3_3D iter 861: spei_z_score_fv3.

SPEI = (P − PET − μ_D)/σ_D.

Tests
-----

1. ``test_at_climatology_zero``: D = μ_D → SPEI = 0.
2. ``test_warming_aridifies``: ↑PET at fixed P → ↓SPEI.
3. ``test_extreme_drought_le_minus2``: D = μ_D − 2σ_D → SPEI = −2.
4. ``test_spei_vs_spi_under_warming``: Same P; SPEI drops more
   negative than SPI when PET rises.
5. ``test_sigma_zero_floored``: σ_D=0 → finite.
6. ``test_monotone_in_precip``: ↑P at fixed PET → ↑SPEI.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    spei_z_score_fv3,
    spi_z_score_fv3,
)


def test_at_climatology_zero():
    """D = μ_D → SPEI = 0."""
    spei = spei_z_score_fv3(
        precip=jnp.array([100.0]),
        pet=jnp.array([50.0]),
        mu_d=jnp.array([50.0]),
        sigma_d=jnp.array([30.0]),
    )
    np.testing.assert_allclose(np.asarray(spei), [0.0], atol=1e-12)


def test_warming_aridifies():
    """↑PET at fixed P → ↓SPEI (drought intensification under warming)."""
    base = spei_z_score_fv3(
        precip=jnp.array([100.0]),
        pet=jnp.array([50.0]),
        mu_d=jnp.array([50.0]),
        sigma_d=jnp.array([30.0]),
    )
    warm = spei_z_score_fv3(
        precip=jnp.array([100.0]),
        pet=jnp.array([80.0]),  # +30 mm PET
        mu_d=jnp.array([50.0]),
        sigma_d=jnp.array([30.0]),
    )
    assert float(warm[0]) < float(base[0])


def test_extreme_drought_le_minus2():
    """D = μ_D − 2σ_D → SPEI = −2."""
    spei = spei_z_score_fv3(
        precip=jnp.array([20.0]),
        pet=jnp.array([30.0]),  # D=−10
        mu_d=jnp.array([50.0]),  # μ=50, so D−μ=−60
        sigma_d=jnp.array([30.0]),  # /30 = −2
    )
    np.testing.assert_allclose(np.asarray(spei), [-2.0], rtol=1e-12)


def test_spei_vs_spi_under_warming():
    """Same P, +PET → SPEI more negative than SPI (Vicente-Serrano)."""
    # SPI: P=80, μ_P=100, σ_P=30 → SPI = (80−100)/30 = −0.67
    spi = spi_z_score_fv3(
        precip=jnp.array([80.0]),
        mu_p=jnp.array([100.0]),
        sigma_p=jnp.array([30.0]),
    )
    # SPEI: same P=80, PET higher than baseline → D=80−60=20
    # Baseline D climatology: μ_D=50, σ_D=30 → SPEI = (20-50)/30 = -1.0
    spei = spei_z_score_fv3(
        precip=jnp.array([80.0]),
        pet=jnp.array([60.0]),
        mu_d=jnp.array([50.0]),
        sigma_d=jnp.array([30.0]),
    )
    # SPEI (−1.0) more drought-intense than SPI (−0.67)
    assert float(spei[0]) < float(spi[0])


def test_sigma_zero_floored():
    """σ_D=0 → finite via floor."""
    spei = spei_z_score_fv3(
        precip=jnp.array([100.0]),
        pet=jnp.array([50.0]),
        mu_d=jnp.array([50.0]),
        sigma_d=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(spei))


def test_monotone_in_precip():
    """↑P at fixed PET → ↑SPEI."""
    pet = jnp.full((4,), 50.0)
    mu = jnp.full((4,), 50.0)
    sig = jnp.full((4,), 30.0)
    p = jnp.array([50.0, 80.0, 120.0, 160.0])
    spei = spei_z_score_fv3(p, pet, mu, sig)
    diffs = jnp.diff(spei)
    assert jnp.all(diffs > 0.0)


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=861)
    n_x, n_y = 6, 8
    p = jnp.asarray(rng.uniform(0.0, 200.0, size=(n_x, n_y)))
    pet = jnp.asarray(rng.uniform(20.0, 150.0, size=(n_x, n_y)))
    mu = jnp.asarray(rng.uniform(0.0, 100.0, size=(n_x, n_y)))
    sig = jnp.asarray(rng.uniform(10.0, 50.0, size=(n_x, n_y)))
    spei = spei_z_score_fv3(p, pet, mu, sig)
    assert spei.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(spei))
