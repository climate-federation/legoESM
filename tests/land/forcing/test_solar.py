"""Unit tests for the shared solar-zenith helper."""

import jax.numpy as jnp
import numpy as np

from legoesm.land.forcing.solar import cos_solar_zenith


def test_night_is_zero():
    # Local midnight at the equator on the equinox -> sun below horizon.
    # lon 0 -> local hour == UTC hour; hour 0 == midnight.
    val = float(cos_solar_zenith(0.0, 0.0, 80.0, 0.0))
    assert val == 0.0


def test_noon_equator_equinox_near_one():
    # Local solar noon at the equator on the equinox -> overhead sun.
    val = float(cos_solar_zenith(0.0, 0.0, 80.0, 12.0))
    assert val > 0.99


def test_bounded_unit_interval():
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 19)
    lon = jnp.linspace(0.0, 2 * jnp.pi, 19)
    for doy in (1.0, 90.0, 180.0, 270.0, 365.0):
        for hour in np.linspace(0.0, 24.0, 13):
            cz = cos_solar_zenith(lat, lon, doy, hour)
            assert jnp.all(cz >= 0.0)
            assert jnp.all(cz <= 1.0 + 1e-6)


def test_longitude_shifts_local_noon():
    # At 90 deg E, local noon occurs 6 h earlier in UTC than at lon 0.
    east = float(cos_solar_zenith(0.0, jnp.pi / 2, 80.0, 6.0))
    assert east > 0.99
