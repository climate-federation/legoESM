"""FV3_3D iter 778: coriolis_parameter_fv3 + dcmip16_tc refactor.

f = 2·Ω·sin(lat), Ω = constants.Omega = 7.292e-5 rad/s.

Refactors inline 2·Ω·sin(lat) pattern in iter-672
``dcmip16_tc_uwind_pert``; dcmip16-TC tests pass post-refactor.

Tests
-----

1. ``test_f_equator``: lat=0 → f=0.
2. ``test_f_north_pole``: lat=π/2 → f = 2·Ω.
3. ``test_f_south_pole``: lat=−π/2 → f = −2·Ω.
4. ``test_f_midlat_known``: lat=40°N → f ≈ 9.376·10⁻⁵ s⁻¹.
5. ``test_f_units_rad_eq_deg``: units='deg' agrees with units='rad'.
6. ``test_f_shapes_3d_finite``.
7. ``test_f_invalid_units``: bad units raises ValueError.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import coriolis_parameter_fv3


def test_f_equator():
    """f(0) = 0."""
    lat = jnp.array([0.0])
    f = coriolis_parameter_fv3(lat)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-14)


def test_f_north_pole():
    """f(π/2) = 2·Ω."""
    lat = jnp.array([jnp.pi / 2.0])
    f = coriolis_parameter_fv3(lat)
    np.testing.assert_allclose(
        np.asarray(f), [2.0 * constants.Omega], atol=1e-14
    )


def test_f_south_pole():
    """f(-π/2) = -2·Ω."""
    lat = jnp.array([-jnp.pi / 2.0])
    f = coriolis_parameter_fv3(lat)
    np.testing.assert_allclose(
        np.asarray(f), [-2.0 * constants.Omega], atol=1e-14
    )


def test_f_midlat_known():
    """Textbook value at 40°N: f ≈ 9.376e-5 s⁻¹."""
    lat = jnp.array([40.0])
    f = coriolis_parameter_fv3(lat, units="deg")
    # 2·7.292e-5·sin(40°) ≈ 9.376e-5
    expected = 2.0 * constants.Omega * np.sin(np.radians(40.0))
    np.testing.assert_allclose(np.asarray(f), [expected], rtol=1e-12)
    # Sanity check absolute value (drift detector)
    assert 9.0e-5 < float(f[0]) < 1.0e-4


def test_f_units_rad_eq_deg():
    """Same lat in both rad and deg gives the same f."""
    rng = np.random.default_rng(seed=778)
    lat_deg = jnp.asarray(rng.uniform(-90.0, 90.0, size=(50,)))
    lat_rad = jnp.radians(lat_deg)
    f_deg = coriolis_parameter_fv3(lat_deg, units="deg")
    f_rad = coriolis_parameter_fv3(lat_rad, units="rad")
    np.testing.assert_allclose(np.asarray(f_deg), np.asarray(f_rad), atol=1e-15)


def test_f_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=779)
    n_x, n_y = 6, 8
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2, jnp.pi / 2, size=(n_x, n_y)))
    f = coriolis_parameter_fv3(lat)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))


def test_f_invalid_units():
    """Bad units string raises ValueError."""
    lat = jnp.array([0.0])
    with pytest.raises(ValueError):
        coriolis_parameter_fv3(lat, units="bogus")
