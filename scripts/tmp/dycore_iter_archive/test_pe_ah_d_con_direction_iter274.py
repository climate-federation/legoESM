"""FV3_3D iter 274: explicit direction test for iter-225
ah_d_con (PE Smagorinsky-A_h d_con) — mean dT_d_con > 0 when
KE is removed.

Closes the PE per-mechanism direction-test trio (iter-272/273
for corner-div / div_damp; iter-274 for A_h).
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_ah_d_con_net_heating():
    """Mean(dT_d_con) > 0 when iter-57/58 A_h Laplacian removes
    KE."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=274)
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
        # Only A_h active to isolate.
        A_h=1e6, smagorinsky_cs=0.20,
        damp_v=0.0,
        corner_div_damp_d2_bg=0.0,
        div_damp_coeff=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(**common, ah_d_con=0.0)
    cfg_on = CDGridPrimitiveEquationConfig(**common, ah_d_con=1.0)

    s_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off).step(
        state, 100.0,
    )
    s_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on).step(
        state, 100.0,
    )

    dT_d_con = s_on.T.data - s_off.T.data
    mean_heating = float(jnp.mean(dT_d_con))

    assert mean_heating > 0.0, (
        f"ah_d_con=1 must produce NET heating; got "
        f"mean(dT_d_con)={mean_heating:.4e} K."
    )

    max_dT = float(jnp.max(jnp.abs(dT_d_con)))
    assert max_dT < 10.0
