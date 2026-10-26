"""Temperature-dependent snow ageing (BATS/CLM grain growth).

Pins: off is byte-identical to the calendar clock; the rate is 1 at freezing,
falls steeply when cold, and is never above 1; cold snow keeps a bright albedo
for months while melting snow darkens on the old timescale.  Each assertion
fails if the temperature scaling is removed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.snow_budget import metamorphism_rate, update_snow_age
from legoesm.surface_albedo import LandAlbedoConfig, snow_albedo

DAY = 86400.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_rate_is_one_at_freezing_and_bounded():
    T = jnp.array([constants.T_freeze, 240.0, 230.0, 200.0, 300.0])
    r = np.asarray(metamorphism_rate(T, 5000.0))
    assert r[0] == pytest.approx(1.0, rel=1e-12)
    assert np.all(r <= 1.0) and np.all(r > 0.0)          # never ages faster
    assert r[1] == pytest.approx(0.0797, abs=2e-3)       # 240 K: ~12x slower
    assert r[2] < r[1] < 1.0                             # colder = slower
    assert r[4] == 1.0                                   # clipped above freezing


def test_activation_zero_is_the_calendar_clock():
    T = jnp.full((4,), 240.0)
    swe = jnp.full((4,), 100.0)
    age = jnp.full((4,), 10 * DAY)
    fresh = jnp.zeros((4,))
    a = update_snow_age(swe, age, fresh, DAY, age_activation_K=0.0)
    b = update_snow_age(swe, age, fresh, DAY, T_snow=T, age_activation_K=0.0)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_cold_snow_ages_slower_than_melting_snow():
    swe = jnp.full((2,), 100.0)
    age = jnp.zeros((2,))
    fresh = jnp.zeros((2,))
    T = jnp.array([230.0, constants.T_freeze])
    out = np.asarray(update_snow_age(swe, age, fresh, DAY, T_snow=T,
                                     age_activation_K=5000.0))
    assert out[1] == pytest.approx(DAY, rel=1e-12)       # melting: full day
    assert out[0] < 0.05 * DAY                           # cold: ~30x slower
    assert out[0] > 0.0


def test_two_months_of_cold_snow_stays_bright():
    """The defect this fixes: 55 days of polar snow pinned at alpha_snow_min."""
    cfg = LandAlbedoConfig(alpha_snow_max=0.8077, alpha_snow_min=0.5207,
                           tau_snow_decay=3.674 * DAY)
    swe = jnp.full((1,), 500.0)
    fresh = jnp.zeros((1,))
    for T_val, expect_bright in ((230.0, True), (constants.T_freeze, False)):
        age = jnp.zeros((1,))
        for _ in range(55):
            age = update_snow_age(swe, age, fresh, DAY,
                                  T_snow=jnp.full((1,), T_val),
                                  age_activation_K=5000.0)
        alb = float(np.asarray(snow_albedo(age, cfg))[0])
        if expect_bright:
            # MEASURED: 0.698 at 230 K over 55 days with BATS A = 5000 K and the
            # run's calibrated ceiling 0.8077.  That is Arctic-tundra bright
            # (observed 0.6-0.75) but still short of the Antarctic plateau
            # (0.80-0.85); closing that needs the CEILING raised too, which the
            # calendar clock never exercised.  Pinned at the measured value so a
            # change to the rate law shows up here.
            assert alb > 0.68, alb
        else:
            assert alb < 0.53, alb          # melting snow reaches the floor
    # and with the feature OFF the cold column is dark, i.e. the test is not vacuous
    age = jnp.zeros((1,))
    for _ in range(55):
        age = update_snow_age(swe, age, fresh, DAY, age_activation_K=0.0)
    assert float(np.asarray(snow_albedo(age, cfg))[0]) < 0.53


def test_fresh_snow_still_dilutes_the_effective_age():
    swe = jnp.full((1,), 20.0)
    age = jnp.full((1,), 100 * DAY)
    fresh = jnp.full((1,), 19.0 / DAY)      # kg/m2/s -> 19 kg/m2 this step
    out = float(np.asarray(update_snow_age(swe, age, fresh, DAY,
                                           T_snow=jnp.full((1,), 230.0),
                                           age_activation_K=5000.0))[0])
    assert out < 10 * DAY                   # mass dilution still dominates


@pytest.mark.parametrize("activation_K", [None, 3000.0])
def test_slab_land_ages_snow_with_the_configured_activation(activation_K):
    """The slab land must read LandConfig.land_albedo.snow_age_activation_K;
    with the calendar clock (0.0) one cold hour ages the pack a full hour."""
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.core.field import Field
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land
    from legoesm.land.state import LandState

    ncol, dt, T0, age0 = 2, 3600.0, 250.0, 10 * DAY
    ones = jnp.ones(ncol)
    forcing = AtmToSurface(**{k: v * ones for k, v in dict(
        sw_down=50.0, lw_down=220.0, precip_total=0.0, precip_snow=0.0,
        T_lowest=265.0, q_lowest=5e-4, u_lowest=3.0, v_lowest=0.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.3, cos_zenith=0.3,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0).items()})
    state = LandState(
        T_soil=Field(jnp.full(ncol, T0), name="T_soil"),
        W_bucket=Field(jnp.full(ncol, 50.0), name="W_bucket"),
        snow_depth=Field(jnp.full(ncol, 100.0), name="snow_depth"),
        snow_age=Field(jnp.full(ncol, age0), name="snow_age"),
    )
    cfg = LandConfig()
    if activation_K is None:                 # the config default
        activation_K = 5000.0
        assert cfg.land_albedo.snow_age_activation_K == activation_K
    else:
        cfg = cfg._replace(land_albedo=cfg.land_albedo._replace(
            snow_age_activation_K=activation_K))
    new, _, _ = step_land(state, forcing, cfg, U_min=1.0, dt=dt)
    grown = np.asarray(new.snow_age.data) - age0
    expected = dt * float(metamorphism_rate(jnp.asarray(T0), activation_K))
    np.testing.assert_allclose(grown, expected, rtol=1e-9)
    assert (grown < 0.5 * dt).all()          # not the calendar clock's full hour
