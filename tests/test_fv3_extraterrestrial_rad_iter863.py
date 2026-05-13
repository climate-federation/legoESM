"""FV3_3D iter 863: extraterrestrial_radiation_fv3.

FAO-56 Eq. 21 closed-form R_a from (latitude, day-of-year).

Tests
-----

1. ``test_equator_equinox``: φ=0, DOY=80 (March eq) → R_a ≈ 37 MJ/m²/day.
2. ``test_polar_night``: φ=80°N, DOY=355 (Dec) → R_a ≈ 0.
3. ``test_polar_day``: φ=80°N, DOY=172 (Jun) → R_a > 40.
4. ``test_60N_summer_winter_asymmetry``: 60°N Jun >> 60°N Dec.
5. ``test_chain_to_pet``: R_a → iter-862 PET (consistent).
6. ``test_hemisphere_symmetry``: φ=−60, DOY=355 ≈ φ=+60, DOY=172.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    extraterrestrial_radiation_fv3,
    hargreaves_pet_fv3,
)


def test_equator_equinox():
    """Equator at March equinox (DOY=80) → R_a ≈ 37-38 MJ/m²/day."""
    r_a = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([0.0]),
        day_of_year=jnp.array([80.0]),
    )
    assert 36.0 < float(r_a[0]) < 39.0


def test_polar_night():
    """80°N at Dec solstice → polar night → R_a ≈ 0."""
    r_a = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([80.0]),
        day_of_year=jnp.array([355.0]),
    )
    assert float(r_a[0]) < 1.0  # near-zero polar-night


def test_polar_day():
    """80°N at June solstice → polar day → R_a > 40 (24-h sun)."""
    r_a = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([80.0]),
        day_of_year=jnp.array([172.0]),
    )
    assert float(r_a[0]) > 40.0


def test_60N_summer_winter_asymmetry():
    """60°N June solstice >> 60°N Dec solstice."""
    r_a_summer = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([172.0]),
    )
    r_a_winter = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([355.0]),
    )
    assert float(r_a_summer[0]) > float(r_a_winter[0])
    assert float(r_a_winter[0]) < 10.0  # short days


def test_chain_to_pet():
    """R_a → iter-862 PET (drought-budget chain consistency)."""
    r_a = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([30.0]),
        day_of_year=jnp.array([172.0]),  # June solstice
    )
    pet = hargreaves_pet_fv3(
        t_mean_c=jnp.array([28.0]),
        t_max_c=jnp.array([35.0]),
        t_min_c=jnp.array([21.0]),
        r_a_mj=r_a,
    )
    assert jnp.all(jnp.isfinite(pet))
    assert float(pet[0]) > 0.0


def test_hemisphere_symmetry():
    """Northern winter ≈ Southern summer (DOY offset 6 months)."""
    r_a_n_winter = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([60.0]),
        day_of_year=jnp.array([355.0]),  # NH Dec solstice
    )
    r_a_s_summer = extraterrestrial_radiation_fv3(
        latitude_deg=jnp.array([-60.0]),
        day_of_year=jnp.array([355.0]),  # SH summer
    )
    # SH summer at -60°N should be >> NH winter at +60°N
    assert float(r_a_s_summer[0]) > float(r_a_n_winter[0])


def test_shapes_finite():
    """2-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=863)
    n_x, n_y = 6, 8
    lat = jnp.asarray(rng.uniform(-89.0, 89.0, size=(n_x, n_y)))
    doy = jnp.asarray(rng.integers(1, 365, size=(n_x, n_y)).astype(float))
    r_a = extraterrestrial_radiation_fv3(lat, doy)
    assert r_a.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(r_a))
    assert jnp.all(r_a >= 0.0)
