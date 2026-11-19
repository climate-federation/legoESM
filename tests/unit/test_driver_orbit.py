"""The run's one orbit (legoesm.driver.orbit) and the insolation it drives."""

import re
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.solar import (
    OrbitalParameters, daily_mean_insolation, earth_orbit,
)
from legoesm.driver.orbit import experiment_orbit, orbit_from_year

jax.config.update("jax_enable_x64", True)

_DRIVER = Path(__file__).resolve().parents[2] / "packages/coupler/legoesm/driver"


def _cfg(**kw):
    base = dict(orbital_insolation=True, orbit_year=None, orbit_eccentricity=None,
                orbit_obliquity_deg=None, orbit_long_perihelion_deg=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _global_annual_mean(orbit, n_lat=4000, days_per_step=0.25):
    # Equal-area latitudes (uniform in sin(lat)) -> plain mean is the area mean.
    mu = (np.arange(n_lat) + 0.5) / n_lat * 2.0 - 1.0
    lat = jnp.asarray(np.arcsin(mu))
    days = np.arange(1.0, 366.0, days_per_step)
    q = jax.vmap(lambda d: daily_mean_insolation(lat, d, orbit=orbit))(jnp.asarray(days))
    return float(jnp.mean(q))


def test_present_day_is_bitwise_earth_orbit_and_off_is_circular():
    assert experiment_orbit(_cfg()) == earth_orbit()
    assert experiment_orbit(_cfg(orbital_insolation=False)) is None


def test_berger_year_2000_reproduces_present_day_constants():
    o = orbit_from_year(2000)
    p = earth_orbit()
    assert abs(o.eccentricity - p.eccentricity) < 1e-4
    assert abs(o.obliquity_deg - p.obliquity_deg) < 1e-2
    assert abs(o.long_perihelion_deg - p.long_perihelion_deg) < 0.1


def test_elements_override_one_by_one_and_year_resolves():
    o = experiment_orbit(_cfg(orbit_obliquity_deg=60.0))
    assert o == earth_orbit()._replace(obliquity_deg=60.0)
    assert experiment_orbit(_cfg(orbit_year=-4050)) == orbit_from_year(-4050)


@pytest.mark.parametrize("kw, match", [
    (dict(orbital_insolation=False, orbit_obliquity_deg=10.0), "orbital_insolation"),
    (dict(orbit_year=1850, orbit_eccentricity=0.0), "both set"),
    (dict(orbit_eccentricity=0.11), "eccentricity"),
    (dict(orbit_obliquity_deg=-1.0), "obliquity"),
    (dict(orbit_long_perihelion_deg=360.0), "perihelion"),
    (dict(orbit_year=2_000_000), "validity"),
    (dict(orbit_year=1850.5), "integer"),
])
def test_contradictory_or_out_of_range_orbit_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        experiment_orbit(_cfg(**kw))


def test_zero_obliquity_circular_orbit_is_equinox_every_day():
    orbit = OrbitalParameters(0.0, 0.0, 0.0)
    lat = jnp.deg2rad(jnp.linspace(-89.0, 89.0, 37))
    for day in (1.0, 100.0, 172.0, 300.0):
        np.testing.assert_allclose(
            daily_mean_insolation(lat, day, orbit=orbit),
            constants.S_0 / np.pi * np.cos(lat), rtol=1e-12, atol=1e-9)


def test_ninety_degree_obliquity_solstice_matches_polar_day_formula():
    orbit = OrbitalParameters(0.0, 90.0, 0.0)
    solstice = 80.5 + 365.0 / 4.0  # solar longitude 90 deg on a circular orbit
    lat = jnp.deg2rad(jnp.linspace(-85.0, 85.0, 35))
    expected = np.where(lat > 0, constants.S_0 * np.sin(lat), 0.0)
    # Polar-day sunset angle is AD-capped at sqrt(2e-7) from pi (solar.py).
    np.testing.assert_allclose(daily_mean_insolation(lat, solstice, orbit=orbit),
                               expected, atol=0.5)


@pytest.mark.parametrize("obliquity", [0.0, 23.44, 60.0, 90.0])
def test_global_annual_mean_independent_of_obliquity(obliquity):
    mean = _global_annual_mean(OrbitalParameters(0.0, obliquity, 0.0))
    assert abs(mean / (constants.S_0 / 4.0) - 1.0) < 2e-4


def test_distance_factor_applied_exactly_once():
    # Annual mean of (a/r)^2 over the orbit is (1 - e^2)^(-1/2): applied
    # twice or never would give a different global-annual mean.
    e = 0.05
    mean = _global_annual_mean(OrbitalParameters(e, 23.44, 102.0))
    assert abs(mean / (constants.S_0 / 4.0 / np.sqrt(1.0 - e * e)) - 1.0) < 2e-4


def test_polar_day_shortfall_is_the_documented_sunset_angle_cap():
    # Known difference vs climlab (max 0.20 W/m2 at obliquity 90): the AD-safe
    # arccos clip (1 - 1e-7) shrinks polar-day insolation by
    # (S0/pi) sqrt(2e-7) (a/r)^2 sin(lat) sin(decl).
    from legoesm.atmosphere.physics.radiation.solar import (
        earth_sun_distance_factor, solar_declination,
    )
    orbit = OrbitalParameters(0.0167, 90.0, 282.9)
    day, lat = 357.0, jnp.deg2rad(-90.0)
    sd = jnp.sin(solar_declination(day, orbit=orbit))
    f = earth_sun_distance_factor(day, orbit)
    exact = constants.S_0 * f * jnp.sin(lat) * sd
    cap = constants.S_0 / np.pi * np.sqrt(2e-7) * f * jnp.sin(lat) * sd
    shortfall = exact - daily_mean_insolation(lat, day, orbit=orbit)
    np.testing.assert_allclose(shortfall, cap, rtol=1e-2)


def test_validate_strict_refuses_bad_year_before_the_run():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="validity"):
        ExperimentConfig(orbital_insolation=True, orbit_year=-2_000_000).validate_strict()


def test_driver_reads_only_the_shared_orbit():
    for f in sorted(_DRIVER.glob("*.py")):
        if f.name == "orbit.py":
            continue
        assert not re.search(r"\bearth_orbit\b", f.read_text()), (
            f"{f.name} builds its own orbit instead of experiment_orbit()")
    assert "land.forcing.solar" not in (_DRIVER / "model_driver.py").read_text()
