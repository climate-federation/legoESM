"""FV3_3D iter 865: solar_zenith_cos_fv3.

cos(θ_s) = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(ω).

Tests
-----

1. ``test_equator_noon_equinox``: φ=0, J=80, t=12 → cos(θ)=1.
2. ``test_equator_midnight_zero``: φ=0, t=0 → cos(θ)=0 clipped.
3. ``test_30N_noon_equinox``: φ=30°, J=80, noon → cos(θ) ≈ 0.866.
4. ``test_60N_dec_noon_low``: φ=60°N, Dec, noon → low cos(θ) <0.2.
5. ``test_polar_night_zero``: 80°N Dec all hours → cos(θ)=0.
6. ``test_polar_day_positive``: 80°N June midnight → cos(θ)>0 still.
7. ``test_sunrise_sunset_zero``: 06:00/18:00 at equator equinox → ~0.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import solar_zenith_cos_fv3


def test_equator_noon_equinox():
    """Equator at March equinox noon → cos(θ) ≈ 1 (sun overhead)."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([0.0]),
        day_of_year=jnp.array([80.0]),
        hour_local=jnp.array([12.0]),
    )
    # δ ≈ 0 at March equinox (DOY 80 close to it); cos(0)=1 sun overhead
    assert float(cos_theta[0]) > 0.99


def test_equator_midnight_zero():
    """Equator midnight → cos(θ)=0 (sun below horizon, clipped)."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([0.0]),
        day_of_year=jnp.array([172.0]),  # June solstice
        hour_local=jnp.array([0.0]),
    )
    np.testing.assert_allclose(np.asarray(cos_theta), [0.0], atol=1e-14)


def test_30N_noon_equinox():
    """30°N March equinox noon → cos(θ) ≈ cos(30°) ≈ 0.866."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([30.0]),
        day_of_year=jnp.array([80.0]),
        hour_local=jnp.array([12.0]),
    )
    assert 0.85 < float(cos_theta[0]) < 0.88


def test_60N_dec_noon_low():
    """60°N Dec solstice noon → low sun, cos(θ) < 0.2."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([355.0]),
        hour_local=jnp.array([12.0]),
    )
    # cos(60° + 23.5°) = cos(83.5°) ≈ 0.113
    assert 0.05 < float(cos_theta[0]) < 0.20


def test_polar_night_zero():
    """80°N Dec solstice → polar night, cos(θ)=0 all hours."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([80.0, 80.0, 80.0]),
        day_of_year=jnp.array([355.0, 355.0, 355.0]),
        hour_local=jnp.array([0.0, 12.0, 18.0]),
    )
    np.testing.assert_allclose(np.asarray(cos_theta), [0, 0, 0], atol=1e-12)


def test_polar_day_positive():
    """80°N June solstice midnight → still cos(θ)>0 (polar day)."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([80.0]),
        day_of_year=jnp.array([172.0]),
        hour_local=jnp.array([0.0]),
    )
    assert float(cos_theta[0]) > 0.0


def test_sunrise_sunset_zero():
    """Equator equinox 06:00 / 18:00 → cos(θ) ≈ 0 (horizon)."""
    cos_dawn = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([0.0]),
        day_of_year=jnp.array([80.0]),
        hour_local=jnp.array([6.0]),
    )
    cos_dusk = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([0.0]),
        day_of_year=jnp.array([80.0]),
        hour_local=jnp.array([18.0]),
    )
    # ω = ±π/2, cos(±π/2)=0; at equinox δ≈0 also so cos(θ)=0
    assert float(cos_dawn[0]) < 0.1
    assert float(cos_dusk[0]) < 0.1


def test_shapes_finite():
    """3-D shapes preserved, finite, in [0, 1]."""
    rng = np.random.default_rng(seed=865)
    n_x, n_y = 6, 8
    lat = jnp.asarray(rng.uniform(-89.0, 89.0, size=(n_x, n_y)))
    doy = jnp.asarray(rng.integers(1, 365, size=(n_x, n_y)).astype(float))
    hr = jnp.asarray(rng.uniform(0.0, 24.0, size=(n_x, n_y)))
    cos_theta = solar_zenith_cos_fv3(lat, doy, hr)
    assert cos_theta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(cos_theta))
    assert jnp.all(cos_theta >= 0.0) and jnp.all(cos_theta <= 1.0 + 1e-10)
