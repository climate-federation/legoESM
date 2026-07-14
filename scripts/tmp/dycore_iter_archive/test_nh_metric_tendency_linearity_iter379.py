"""FV3_3D iter 379: NH counterpart of iter-378.

Tendency-level bit-for-bit linearity for the 3 NH slow-tendency
d_con sites under metric form (iter-348/350/352).  Uses
``cdgrid_compressible_euler_slow_tendencies`` directly to bypass
acoustic feedback.

Tests
-----

1. ``test_nh_metric_corner_div_d_con_linear_tendency``
2. ``test_nh_metric_div_damp_d_con_linear_tendency``
3. ``test_nh_metric_ah_d_con_linear_tendency``
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


@pytest.fixture(scope="module")
def state_setup():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=379)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, cdgrid, height_coord, terrain_metric, state


def _check(make_cfg, state, grid, hc, tm, cdgrid):
    cfg_05 = make_cfg(0.5)
    cfg_10 = make_cfg(1.0)
    cfg_20 = make_cfg(2.0)
    dt = 10.0
    t_05 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_05, dt_actual=dt,
    )
    t_10 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_10, dt_actual=dt,
    )
    t_20 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_20, dt_actual=dt,
    )
    d10 = (
        np.asarray(t_10.dtheta_prime_dt.data)
        - np.asarray(t_05.dtheta_prime_dt.data)
    )
    d20 = (
        np.asarray(t_20.dtheta_prime_dt.data)
        - np.asarray(t_05.dtheta_prime_dt.data)
    )
    np.testing.assert_allclose(d20, 3.0 * d10, rtol=1e-10, atol=1e-12)
    assert float(np.max(np.abs(d10))) > 1e-12


def test_nh_metric_corner_div_d_con_linear_tendency(state_setup):
    grid, cdgrid, hc, tm, state = state_setup

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            corner_div_damp_d2_bg=0.0005,
            corner_div_damp_dddmp=0.20,
            corner_div_damp_d_con=d,
            use_fv3_metric_aware_d_con=True,
        )

    _check(make_cfg, state, grid, hc, tm, cdgrid)


def test_nh_metric_div_damp_d_con_linear_tendency(state_setup):
    grid, cdgrid, hc, tm, state = state_setup

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            div_damp_coeff=1e6, div_damp_dddmp=0.20,
            div_damp_d_con=d,
            use_fv3_metric_aware_d_con=True,
        )

    _check(make_cfg, state, grid, hc, tm, cdgrid)


def test_nh_metric_ah_d_con_linear_tendency(state_setup):
    grid, cdgrid, hc, tm, state = state_setup

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            A_h=1e6, ah_d_con=d,
            use_fv3_metric_aware_d_con=True,
        )

    _check(make_cfg, state, grid, hc, tm, cdgrid)
