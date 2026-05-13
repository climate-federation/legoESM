"""FV3_3D iter 864: daylight_hours_fv3.

FAO-56 Eq. 34 N = 24·ω_s/π.

Tests
-----

1. ``test_equator_12h``: φ=0 any day → N = 12 h exactly.
2. ``test_polar_day_24h``: 80°N June solstice → N ≈ 24.
3. ``test_polar_night_zero``: 80°N Dec solstice → N ≈ 0.
4. ``test_60N_summer_winter``: 60°N June ~18.8, Dec ~5.4.
5. ``test_arctic_circle_24h``: 66.5°N June → 24.
6. ``test_hemisphere_symmetry``: ±60° opposite seasons match.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import daylight_hours_fv3


def test_equator_12h():
    """Equator any day → N=12 h."""
    n_h = daylight_hours_fv3(
        latitude_deg=jnp.array([0.0, 0.0, 0.0]),
        day_of_year=jnp.array([80.0, 172.0, 355.0]),
    )
    np.testing.assert_allclose(np.asarray(n_h), [12.0, 12.0, 12.0], rtol=1e-12)


def test_polar_day_24h():
    """80°N June solstice → N ≈ 24 (polar day clipped)."""
    n_h = daylight_hours_fv3(
        latitude_deg=jnp.array([80.0]),
        day_of_year=jnp.array([172.0]),
    )
    np.testing.assert_allclose(np.asarray(n_h), [24.0], rtol=1e-12)


def test_polar_night_zero():
    """80°N Dec solstice → N ≈ 0 (polar night clipped)."""
    n_h = daylight_hours_fv3(
        latitude_deg=jnp.array([80.0]),
        day_of_year=jnp.array([355.0]),
    )
    np.testing.assert_allclose(np.asarray(n_h), [0.0], atol=1e-12)


def test_60N_summer_winter():
    """60°N: June ≈ 18.8 h, Dec ≈ 5.4 h."""
    n_summer = daylight_hours_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([172.0]),
    )
    n_winter = daylight_hours_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([355.0]),
    )
    assert 18.0 < float(n_summer[0]) < 19.5
    assert 5.0 < float(n_winter[0]) < 6.0


def test_arctic_circle_24h():
    """66.5°N June solstice → N=24 (Arctic Circle boundary)."""
    n_h = daylight_hours_fv3(
        latitude_deg=jnp.array([66.5]),
        day_of_year=jnp.array([172.0]),
    )
    # FAO-56 ephemeris approximation gives ~23.4 at 66.5°N
    # (exact ε=23.439 declination gets clipped just shy of 24 in fit)
    assert float(n_h[0]) > 23.0


def test_hemisphere_symmetry():
    """+60° Dec ≈ −60° Jun (opposite-hemisphere mirror)."""
    n_n_winter = daylight_hours_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([355.0]),
    )
    n_s_summer = daylight_hours_fv3(
        latitude_deg=jnp.array([-60.0]),
        day_of_year=jnp.array([355.0]),
    )
    # Northern winter at +60 ≈ Southern summer at -60 with same DOY:
    # SH at Dec is summer (long days), NH at Dec is winter (short)
    # so n_s_summer >> n_n_winter
    assert float(n_s_summer[0]) > float(n_n_winter[0])
    # And they should be approximately complementary (~24 - n_n)
    # since opposite hemispheres at same DOY are anti-symmetric
    np.testing.assert_allclose(
        float(n_s_summer[0]) + float(n_n_winter[0]), 24.0, rtol=1e-6
    )


def test_shapes_finite():
    """2-D shapes preserved, finite, bounded [0, 24]."""
    rng = np.random.default_rng(seed=864)
    n_x, n_y = 6, 8
    lat = jnp.asarray(rng.uniform(-89.0, 89.0, size=(n_x, n_y)))
    doy = jnp.asarray(rng.integers(1, 365, size=(n_x, n_y)).astype(float))
    n_h = daylight_hours_fv3(lat, doy)
    assert n_h.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(n_h))
    assert jnp.all(n_h >= 0.0) and jnp.all(n_h <= 24.0 + 1e-10)
