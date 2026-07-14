"""FV3_3D iter 272: explicit direction test for iter-221
corner_div_damp_d_con (PE).

iter-221 verified d_con > 0 changes T (vs d_con=0).  iter-228
verified the formula bit-for-bit (sign included).  iter-272
adds an EXPLICIT mean-direction sanity check: with corner-div
damping ON and removing KE, the d_con > 0 contribution to T
must produce NET HEATING (mean(dT_d_con) > 0).

Tests
-----

1. ``test_pe_corner_div_d_con_net_heating`` — strong wind IC,
   single PE step, mean(dT_d_con) > 0 (net heating from KE
   removal).
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


def test_pe_corner_div_d_con_net_heating():
    """Mean(dT_d_con) > 0 when corner-div damping is removing
    KE — d_con converts KE→heat globally."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=272)
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
        # Only corner-div damping active to isolate.
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        damp_v=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=1.0,
    )

    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)

    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)

    dT_d_con = s_on.T.data - s_off.T.data
    mean_heating = float(jnp.mean(dT_d_con))

    assert mean_heating > 0.0, (
        f"corner_div_damp_d_con=1 must produce NET heating "
        f"(mean(dT_d_con) > 0); got mean_heating="
        f"{mean_heating:.4e} K.  KE-to-heat conversion sign "
        f"may be wrong in iter-221 wiring."
    )

    # Sanity: bounded magnitude.
    max_dT = float(jnp.max(jnp.abs(dT_d_con)))
    assert max_dT < 10.0, (
        f"Per-cell |dT_d_con|={max_dT:.4e} too large — "
        f"possible sign error or cap miscalibration."
    )
