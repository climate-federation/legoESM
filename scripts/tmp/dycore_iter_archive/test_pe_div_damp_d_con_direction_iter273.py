"""FV3_3D iter 273: explicit direction test for iter-223
div_damp_d_con (PE) — mean dT_d_con > 0 when KE is removed.

Mirrors iter-272's pattern for the iter-221 corner-div d_con.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_div_damp_d_con_net_heating():
    """Mean(dT_d_con) > 0 when iter-5 cell-centre div_damp is
    removing KE."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=273)
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
        # Only div_damp active to isolate.
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.0,
        corner_div_damp_d2_bg=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, div_damp_d_con=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        **common, div_damp_d_con=1.0,
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
        f"div_damp_d_con=1 must produce NET heating; got "
        f"mean(dT_d_con)={mean_heating:.4e} K."
    )

    max_dT = float(jnp.max(jnp.abs(dT_d_con)))
    assert max_dT < 10.0
