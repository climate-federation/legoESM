"""Unit tests for the RCEMIP-LAND slab surface energy update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.idealized.land_rce import (  # noqa: E402
    update_land_slab_temperature,
)


def test_land_slab_energy_balance_drives_temperature_and_conserves_energy():
    T_s = jnp.asarray([[300.0, 301.0]], dtype=jnp.float64)
    sw_down = jnp.asarray([[420.0, 420.0]], dtype=jnp.float64)
    lw_down = jnp.asarray([[360.0, 360.0]], dtype=jnp.float64)
    shflx = jnp.asarray([[35.0, 45.0]], dtype=jnp.float64)
    lhflx = jnp.asarray([[90.0, 100.0]], dtype=jnp.float64)
    dt = 60.0
    heat_capacity = 5.0e5
    albedo = 0.20
    emissivity = 1.0

    T_new, diag = update_land_slab_temperature(
        T_s, sw_down, lw_down, shflx, lhflx,
        dt, heat_capacity, albedo, emissivity,
    )

    expected_net = (
        sw_down * (1.0 - albedo)
        + lw_down
        - emissivity * constants.sigma_sb * T_s ** 4
        - shflx
        - lhflx
    )
    np.testing.assert_allclose(np.asarray(diag["Q_slab"]), np.asarray(expected_net))
    np.testing.assert_allclose(
        np.asarray((T_new - T_s) * heat_capacity),
        np.asarray(expected_net * dt),
    )
    assert float(jnp.mean(T_new - T_s)) > 0.0


def test_land_slab_energy_balance_equilibrium_is_fixed_point():
    T_s = jnp.asarray([[300.0]], dtype=jnp.float64)
    sw_down = jnp.asarray([[400.0]], dtype=jnp.float64)
    shflx = jnp.asarray([[50.0]], dtype=jnp.float64)
    lhflx = jnp.asarray([[110.0]], dtype=jnp.float64)
    albedo = 0.25
    emissivity = 1.0
    lw_down = (
        emissivity * constants.sigma_sb * T_s ** 4
        + shflx + lhflx
        - sw_down * (1.0 - albedo)
    )

    T_new, diag = update_land_slab_temperature(
        T_s, sw_down, lw_down, shflx, lhflx,
        dt=120.0, heat_capacity=2.0e5,
        albedo=albedo, emissivity=emissivity,
    )

    np.testing.assert_allclose(np.asarray(diag["Q_slab"]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(T_new), np.asarray(T_s), atol=1e-12)
