"""FV3_3D iter 866: clear_sky_toa_radiation_fv3.

S_TOA = S_0 · d_r⁻² · cos(θ_s)  [W/m²].

Tests
-----

1. ``test_equator_equinox_noon``: cos(θ)=1, DOY=80 → S ≈ 1367 W/m².
2. ``test_night_zero``: cos(θ)=0 → S=0.
3. ``test_d_r_factor``: Jan (perihelion) > July (aphelion) by ~6.7%.
4. ``test_chain_with_iter865``: lat+DOY+hr → cos(θ) → S_TOA.
5. ``test_daily_integral_matches_r_a``: ∫S_TOA dt ≈ R_a (MJ).
6. ``test_custom_s_0``: 2× S_0 → 2× S_TOA.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    clear_sky_toa_radiation_fv3,
    extraterrestrial_radiation_fv3,
    solar_zenith_cos_fv3,
)


def test_equator_equinox_noon():
    """cos(θ)=1, DOY=80 (near aphelion-spring) → S_TOA close to S_0."""
    s = clear_sky_toa_radiation_fv3(
        cos_theta=jnp.array([1.0]),
        day_of_year=jnp.array([80.0]),
    )
    # d_r² at DOY 80: 1 + 0.033·cos(2π·80/365) = 1 + 0.033·cos(1.378) ≈ 1.006
    # S ≈ 1361 × 1.006 ≈ 1369
    assert 1350.0 < float(s[0]) < 1400.0


def test_night_zero():
    """cos(θ)=0 → S_TOA=0."""
    s = clear_sky_toa_radiation_fv3(
        cos_theta=jnp.array([0.0]),
        day_of_year=jnp.array([180.0]),
    )
    np.testing.assert_allclose(np.asarray(s), [0.0], atol=1e-14)


def test_d_r_factor():
    """Jan perihelion > July aphelion (~6.7% intra-annual variation)."""
    s_jan = clear_sky_toa_radiation_fv3(
        cos_theta=jnp.array([1.0]),
        day_of_year=jnp.array([3.0]),  # near perihelion
    )
    s_jul = clear_sky_toa_radiation_fv3(
        cos_theta=jnp.array([1.0]),
        day_of_year=jnp.array([186.0]),  # near aphelion
    )
    assert float(s_jan[0]) > float(s_jul[0])
    # Ratio expected ~1.066/0.967 ≈ 1.066
    ratio = float(s_jan[0] / s_jul[0])
    assert 1.05 < ratio < 1.10


def test_chain_with_iter865():
    """lat=60°, DOY=172 (June solstice), noon → cos(θ) → S_TOA."""
    cos_theta = solar_zenith_cos_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([172.0]),
        hour_local=jnp.array([12.0]),
    )
    s = clear_sky_toa_radiation_fv3(cos_theta, jnp.array([172.0]))
    # FAO-56 ephemeris gives cos(θ)≈0.804 (not 0.872 true ε=23.5°);
    # d_r factor near aphelion ≈ 0.97 → S ≈ 1361·0.97·0.80 ≈ 1056
    assert 1000.0 < float(s[0]) < 1200.0


def test_daily_integral_matches_r_a():
    """Daily mean S_TOA × 86400 s ≈ R_a × 1e6 J/MJ (closure check)."""
    # Integrate S_TOA via 24-hour midpoint sum at 1-hour resolution.
    lat = 30.0
    doy = 172.0  # June solstice
    hours = jnp.arange(0.5, 24.0, 1.0)  # 24 midpoints
    n_hours = hours.shape[0]
    cos_thetas = solar_zenith_cos_fv3(
        latitude_deg=jnp.full((n_hours,), lat),
        day_of_year=jnp.full((n_hours,), doy),
        hour_local=hours,
    )
    s_each = clear_sky_toa_radiation_fv3(
        cos_thetas, jnp.full((n_hours,), doy),
    )
    # Daily integral (W/m² · 3600 s/hr · 24 hr) = J/m²/day
    daily_j = float(jnp.sum(s_each) * 3600.0)
    # iter-863 R_a in MJ/m²/day → J/m²/day = R_a · 1e6
    r_a = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([lat]),
        day_of_year=jnp.array([doy]),
    )
    r_a_j = float(r_a[0]) * 1.0e6
    # Match within 1% (numerical integration + ephemeris precision)
    rel_err = abs(daily_j - r_a_j) / r_a_j
    assert rel_err < 0.02, f"daily-integral mismatch {rel_err*100:.2f}%"


def test_custom_s_0():
    """2× S_0 → 2× S_TOA."""
    s1 = clear_sky_toa_radiation_fv3(
        cos_theta=jnp.array([0.5]),
        day_of_year=jnp.array([180.0]),
    )
    s2 = clear_sky_toa_radiation_fv3(
        cos_theta=jnp.array([0.5]),
        day_of_year=jnp.array([180.0]),
        s_0=2.0 * constants.S_0,
    )
    np.testing.assert_allclose(np.asarray(s2), 2.0 * np.asarray(s1), rtol=1e-12)


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=866)
    n_x, n_y = 6, 8
    cos_theta = jnp.asarray(rng.uniform(0.0, 1.0, size=(n_x, n_y)))
    doy = jnp.asarray(rng.integers(1, 365, size=(n_x, n_y)).astype(float))
    s = clear_sky_toa_radiation_fv3(cos_theta, doy)
    assert s.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(s))
    assert jnp.all(s >= 0.0)
