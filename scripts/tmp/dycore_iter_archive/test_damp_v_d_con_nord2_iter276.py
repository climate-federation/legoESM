"""FV3_3D iter 276: damp_v_d_con direction test with FV3
production nord_v=2 (del-6 vorticity damping).

iter-208/209 direction tests use nord_v=1 (or 0).  FV3
production default is nord_v=2.  This iter explicitly
verifies the damp_v_d_con direction property holds with
production-default nord.

Tests
-----

1. ``test_pe_damp_v_d_con_nord2_net_heating`` — PE with
   nord_v=2: mean(dT_d_con) > 0 when damp_v removes KE.
2. ``test_nh_damp_v_d_con_nord2_net_heating`` — NH same.
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
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def test_pe_damp_v_d_con_nord2_net_heating():
    """PE damp_v_d_con with FV3 production nord_v=2."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=276)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    common = dict(
        damp_v=0.030, nord_v=2,    # FV3 production
        corner_div_damp_d2_bg=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=1.0,
    )
    s_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off).step(
        state, 100.0,
    )
    s_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on).step(
        state, 100.0,
    )

    dT_d_con = s_on.T.data - s_off.T.data
    mean_heating = float(jnp.mean(dT_d_con))
    assert mean_heating > 0.0, (
        f"PE damp_v_d_con nord_v=2: mean(dT_d_con)="
        f"{mean_heating:.4e} not > 0."
    )


def test_nh_damp_v_d_con_nord2_net_heating():
    """NH damp_v_d_con with FV3 production nord_v=2."""
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

    rng = np.random.default_rng(seed=2760)
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

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=2,    # FV3 production
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=0.0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0,
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
        f"NH damp_v_d_con nord_v=2: mean(dT_eq)="
        f"{mean_heating:.4e} not > 0."
    )
