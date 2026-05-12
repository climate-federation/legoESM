"""FV3_3D iter 780: beta_plane_fv3 (β = df/dy).

β = 2·Ω·cos(lat) / R_earth.

Tests
-----

1. ``test_beta_equator_max``: lat=0 → β = 2·Ω/R (max).
2. ``test_beta_north_pole_zero``: lat=π/2 → β = 0.
3. ``test_beta_south_pole_zero``: lat=-π/2 → β = 0.
4. ``test_beta_textbook_midlat``: lat=45° → β ≈ 1.62·10⁻¹¹.
5. ``test_beta_hemisphere_symmetry``: β(+lat) = β(-lat).
6. ``test_beta_units_rad_eq_deg``: 'rad' and 'deg' agree.
7. ``test_beta_invalid_units``: bad units string raises ValueError.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import beta_plane_fv3


def test_beta_equator_max():
    """At equator: β = 2·Ω/R (cos=1)."""
    lat = jnp.array([0.0])
    beta = beta_plane_fv3(lat)
    expected = 2.0 * constants.Omega / constants.R_earth
    np.testing.assert_allclose(np.asarray(beta), [expected], rtol=1e-12)
    # Textbook: ~2.29e-11 s⁻¹m⁻¹
    assert 2.0e-11 < float(beta[0]) < 2.5e-11


def test_beta_north_pole_zero():
    """At north pole: β = 0 (cos(π/2) = 0)."""
    lat = jnp.array([jnp.pi / 2.0])
    beta = beta_plane_fv3(lat)
    np.testing.assert_allclose(np.asarray(beta), [0.0], atol=1e-15)


def test_beta_south_pole_zero():
    """At south pole: β = 0."""
    lat = jnp.array([-jnp.pi / 2.0])
    beta = beta_plane_fv3(lat)
    np.testing.assert_allclose(np.asarray(beta), [0.0], atol=1e-15)


def test_beta_textbook_midlat():
    """45°N: β ≈ 1.62·10⁻¹¹ s⁻¹m⁻¹ (textbook value)."""
    lat = jnp.array([45.0])
    beta = beta_plane_fv3(lat, units="deg")
    expected = 2.0 * constants.Omega * np.cos(np.radians(45.0)) / constants.R_earth
    np.testing.assert_allclose(np.asarray(beta), [expected], rtol=1e-12)
    assert 1.5e-11 < float(beta[0]) < 1.7e-11


def test_beta_hemisphere_symmetry():
    """β(+lat) = β(-lat) since cos is even."""
    lat_pos = jnp.array([10.0, 30.0, 50.0, 70.0])
    lat_neg = -lat_pos
    b_pos = beta_plane_fv3(lat_pos, units="deg")
    b_neg = beta_plane_fv3(lat_neg, units="deg")
    np.testing.assert_allclose(np.asarray(b_pos), np.asarray(b_neg), atol=1e-15)


def test_beta_units_rad_eq_deg():
    """Same lat in 'rad' and 'deg' gives same β."""
    rng = np.random.default_rng(seed=780)
    lat_deg = jnp.asarray(rng.uniform(-90.0, 90.0, size=(50,)))
    lat_rad = jnp.radians(lat_deg)
    b_deg = beta_plane_fv3(lat_deg, units="deg")
    b_rad = beta_plane_fv3(lat_rad, units="rad")
    np.testing.assert_allclose(np.asarray(b_deg), np.asarray(b_rad), atol=1e-18)


def test_beta_invalid_units():
    """Invalid units string raises ValueError."""
    lat = jnp.array([0.0])
    with pytest.raises(ValueError):
        beta_plane_fv3(lat, units="bogus")
