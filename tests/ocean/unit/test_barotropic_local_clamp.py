"""SOTA-local split-explicit barotropic (MOM6/MPAS-Ocean style).

``barotropic_local_subcycle_clamp=True`` makes the per-substep eta-floor clamp
LOCAL (jnp.maximum, NO allreduce) and defers the global mass-conserving
redistribute to ONCE per outer step (on the time-averaged eta).  In a deep ocean
with no wetting/drying, NO cell ever hits eta_floor, so BOTH the local
jnp.maximum and the per-substep redistribute are no-ops — the two paths are
BIT-IDENTICAL.  The win is purely the collective count: ~n_substeps allreduces/
step -> 1 (the multi-node strong-scaling lever; the implicit_cn analogue is the
120-allreduce wall).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)


def _state(grid, z_coord):
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(7)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.05 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.05 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.01 * rng.standard_normal((n_lat, n_lon))  # << |eta_floor| ~ 4000 m
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)))


def test_local_clamp_bit_identical_deep_ocean():
    grid = create_latlon_grid(n_lat=36, n_lon=72)
    z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
    state = _state(grid, z_coord)

    cfg_off = LatLonCGridOceanConfig.from_flat(bottom_drag_r=0.0)
    cfg_on = LatLonCGridOceanConfig.from_flat(
        bottom_drag_r=0.0, barotropic_local_subcycle_clamp=True)
    assert cfg_off.barotropic_local_subcycle_clamp is False
    assert cfg_on.barotropic_local_subcycle_clamp is True

    out_off = barotropic_substeps_latlon_cgrid(state, 30.0, 12, grid, z_coord, cfg_off)
    out_on = barotropic_substeps_latlon_cgrid(state, 30.0, 12, grid, z_coord, cfg_on)
    s_off, (hu_off, hv_off) = out_off
    s_on, (hu_on, hv_on) = out_on

    # Deep ocean: no cell hits eta_floor, so local clamp == redistribute (both
    # no-ops) -> every output is bit-identical.
    np.testing.assert_array_equal(
        np.asarray(s_on.eta.data), np.asarray(s_off.eta.data),
        err_msg="eta differs: local clamp not a no-op in deep ocean")
    np.testing.assert_array_equal(
        np.asarray(s_on.u.data), np.asarray(s_off.u.data),
        err_msg="u differs")
    np.testing.assert_array_equal(
        np.asarray(s_on.v.data), np.asarray(s_off.v.data),
        err_msg="v differs")
    np.testing.assert_array_equal(np.asarray(hu_on), np.asarray(hu_off))
    np.testing.assert_array_equal(np.asarray(hv_on), np.asarray(hv_off))
