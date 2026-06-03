"""Outer baroclinic momentum SSP-RK3 integrator (NEMO key_RK3 mirror).

Config-gated ``LatLonCGridOceanConfig.momentum_time_integrator``:
- validation (rejects unknown),
- a step with ``"rk3"`` produces finite output,
- the default ``"euler"`` is bit-exact to an explicitly-constructed euler
  config (the gate is a static Python branch; default preserves regression).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


@pytest.fixture
def pieces():
    grid = create_latlon_grid(n_lat=20, n_lon=40)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0,
                                  dz_surface=10.0, dz_deep=500.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    # HORIZONTALLY-VARYING baroclinic front so the pressure-gradient ->
    # momentum tendency is non-trivial (a horizontally-uniform perturbation
    # gives zero PGF -> zero velocity -> RK3 == euler trivially).  Use a
    # grid-agnostic meridional (j-index) front so the test does not depend
    # on the grid's coordinate-attribute names.
    T = np.asarray(state.T.data)
    nlat = T.shape[0]
    front = np.tanh((np.arange(nlat) / nlat - 0.5) * 6.0)   # (n_lat,)
    zprof = np.exp(-(np.linspace(0, 1, T.shape[-1])))        # surface-intensified
    T = T + 3.0 * front[:, None, None] * zprof[None, None, :]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    return grid, z_coord, state


def test_default_is_euler(pieces):
    assert LatLonCGridOceanConfig().momentum_time_integrator == "euler"


def test_invalid_rejected(pieces):
    grid, z_coord, _ = pieces
    cfg = LatLonCGridOceanConfig(momentum_time_integrator="midpoint")
    with pytest.raises(ValueError, match="momentum_time_integrator"):
        LatLonCGridOceanModel(grid, z_coord, cfg)


def test_rk3_finite_output(pieces):
    grid, z_coord, state = pieces
    model = LatLonCGridOceanModel(
        grid, z_coord, LatLonCGridOceanConfig(momentum_time_integrator="rk3"))
    s = state
    for _ in range(5):
        s = model.step(s, dt=600.0)
    assert bool(jnp.all(jnp.isfinite(s.u.data)))
    assert bool(jnp.all(jnp.isfinite(s.v.data)))
    assert bool(jnp.all(jnp.isfinite(s.T.data)))


def test_euler_default_bit_exact(pieces):
    # default (euler) must reproduce an explicit euler config bit-for-bit
    grid, z_coord, state = pieces
    base = LatLonCGridOceanModel(grid, z_coord, LatLonCGridOceanConfig())
    eul = LatLonCGridOceanModel(
        grid, z_coord, LatLonCGridOceanConfig(momentum_time_integrator="euler"))
    sb = base.step(state, dt=600.0)
    se = eul.step(state, dt=600.0)
    np.testing.assert_array_equal(np.asarray(sb.u.data), np.asarray(se.u.data))
    np.testing.assert_array_equal(np.asarray(sb.v.data), np.asarray(se.v.data))


def test_rk3_differs_from_euler(pieces):
    # rk3 must actually change the momentum update (not silently == euler)
    grid, z_coord, state = pieces
    eul = LatLonCGridOceanModel(grid, z_coord, LatLonCGridOceanConfig())
    rk3 = LatLonCGridOceanModel(
        grid, z_coord, LatLonCGridOceanConfig(momentum_time_integrator="rk3"))
    se = eul.step(state, dt=600.0)
    sr = rk3.step(state, dt=600.0)
    # RK3 differs from euler at O(dt^2); on a gentle idealized case the
    # difference is tiny (~1e-13) — assert it is non-zero (not a silent
    # no-op), not that it exceeds an allclose tolerance.  (At the eORCA1
    # cold-start scale the difference is large: 87 vs 111 m/s by step 40.)
    assert not np.array_equal(np.asarray(se.u.data), np.asarray(sr.u.data))
