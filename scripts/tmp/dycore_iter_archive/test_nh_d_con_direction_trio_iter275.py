"""FV3_3D iter 275: NH counterparts of iter-272/273/274 — explicit
direction tests for the 3 NH tendency-based d_con sites
(corner-div, cell-centre div_damp, A_h Smagorinsky).

For each mechanism, mean(dθ_p_d_con) > 0 when the mechanism is
removing KE (FV3-faithful KE→heat conversion).  Mirrors the
PE direction-trio iter-272/273/274.

Tests
-----

1. ``test_nh_corner_div_damp_d_con_net_heating`` — iter-222.
2. ``test_nh_div_damp_d_con_net_heating`` — iter-224.
3. ``test_nh_ah_d_con_net_heating`` — iter-226.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


def _state(seed=275):
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def _check_net_heating(make_cfg_off, make_cfg_on, seed):
    grid, hc, tm, state = _state(seed=seed)
    s_off = CDGridCompressibleEulerModel(
        grid, hc, tm, make_cfg_off(),
    ).step(state, 10.0)
    s_on = CDGridCompressibleEulerModel(
        grid, hc, tm, make_cfg_on(),
    ).step(state, 10.0)

    dtheta_d_con = (
        s_on.theta_prime.data - s_off.theta_prime.data
    )
    # Convert to T-equivalent for direction check.
    exner_ref = hc.exner_ref
    dT_eq = dtheta_d_con * exner_ref[None, None, None, :]
    mean_heating = float(jnp.mean(dT_eq))
    assert mean_heating > 0.0, (
        f"NH d_con must produce NET heating; got mean(dT_eq)="
        f"{mean_heating:.4e} K."
    )


def test_nh_corner_div_damp_d_con_net_heating():
    """iter-222 NH corner-div d_con net heating."""
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
    )
    _check_net_heating(
        lambda: CDGridCompressibleEulerConfig(
            **common, corner_div_damp_d_con=0.0,
        ),
        lambda: CDGridCompressibleEulerConfig(
            **common, corner_div_damp_d_con=1.0,
        ),
        seed=275,
    )


def test_nh_div_damp_d_con_net_heating():
    """iter-224 NH cell-centre div_damp d_con net heating."""
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
    )
    _check_net_heating(
        lambda: CDGridCompressibleEulerConfig(
            **common, div_damp_d_con=0.0,
        ),
        lambda: CDGridCompressibleEulerConfig(
            **common, div_damp_d_con=1.0,
        ),
        seed=2750,
    )


def test_nh_ah_d_con_net_heating():
    """iter-226 NH Smagorinsky-A_h d_con net heating."""
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    _check_net_heating(
        lambda: CDGridCompressibleEulerConfig(
            **common, ah_d_con=0.0,
        ),
        lambda: CDGridCompressibleEulerConfig(
            **common, ah_d_con=1.0,
        ),
        seed=2751,
    )
