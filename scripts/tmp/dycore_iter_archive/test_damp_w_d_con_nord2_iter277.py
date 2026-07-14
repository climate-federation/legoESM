"""FV3_3D iter 277: damp_w_d_con direction test with FV3
production default nord_w=2 (del-6 w damping).

iter-203 direction test uses nord_w=1.  iter-277 adds the
nord_w=2 case to mirror iter-276's PE damp_v_d_con nord_v=2.

Tests
-----

1. ``test_nh_damp_w_d_con_nord2_net_heating`` — w
   perturbation (respecting w=0 BC) + damp_w + damp_w_d_con,
   mean(dT_eq) > 0 with nord_w=2.
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


def test_nh_damp_w_d_con_nord2_net_heating():
    """NH damp_w_d_con with FV3 production nord_w=2."""
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

    rng = np.random.default_rng(seed=277)
    # w-perturbation respecting w=0 BC at boundaries.
    w_pert = np.zeros((6, n, n, nlev + 1))
    w_pert[..., 1:nlev] = rng.uniform(
        -2.0, 2.0, size=(6, n, n, nlev - 1),
    )

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_pert), name="w",
                dims=dims_w, units="m/s"),
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

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=2,    # FV3 production
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=0.0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=1.0,
    )

    s_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    ).step(state, 10.0)
    s_on = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_on,
    ).step(state, 10.0)

    dtheta_d_con = (
        s_on.theta_prime.data - s_off.theta_prime.data
    )
    dT_eq = dtheta_d_con * height_coord.exner_ref[None, None, None, :]
    mean_heating = float(jnp.mean(dT_eq))
    assert mean_heating > 0.0, (
        f"NH damp_w_d_con nord_w=2: mean(dT_eq)="
        f"{mean_heating:.4e} not > 0."
    )
