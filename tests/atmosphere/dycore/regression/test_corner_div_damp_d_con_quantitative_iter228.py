"""FV3_3D iter 228: quantitative formula test for the iter-221
corner-div damping d_con (PE).

iter-205 / iter-211 verified that the damp_w / damp_v d_con heat
formulas match the expected energy-conservation budget.  iter 228
mirrors the pattern for iter-221 corner-div d_con.

The corner-div damping d_con is added INSIDE the tendency
function (not a post-step block), so the cleanest validation is
to call ``fv3_hydrostatic_tendencies`` directly and compare the
dT_dt output for d_con=0 vs d_con=1.

Expected formula (per second, leading order in dt):

    dT/dt = -corner_div_damp_d_con
            * project_to_cc(u_d * du_d_dt_cdd + v_d * dv_d_dt_cdd)
            / c_pd

where
    du_d_dt_cdd = -∂x(damp * delpc) / 2dx_corner
    dv_d_dt_cdd = -∂y(damp * delpc) / 2dy_corner

Tests
-----

1. ``test_corner_div_damp_d_con_dT_dt_matches_formula`` — at a
   non-rest state, the dT_dt difference between d_con=1 and
   d_con=0 runs equals the expected formula evaluated on the
   exact same state.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.operators_cdgrid import interp_corner_to_center
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_corner_div_damp_d_con_dT_dt_matches_formula():
    """The d_con contribution to dT_dt matches the expected
    formula for the corner-div damping tendency."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    # Strong winds to make the corner-div tendency non-trivial.
    rng = np.random.default_rng(seed=228)
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
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d_con=1.0,
    )

    dt = 100.0
    tend_off = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_off, dt_actual=dt,
    )
    tend_on = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_on, dt_actual=dt,
    )

    # The d_con contribution to dT_dt is the difference.
    dT_dt_d_con = tend_on.dT_dt.data - tend_off.dT_dt.data

    # Expected formula: replicate the iter-221 wiring at the test
    # level.  We need (du_d_dt_cdd, dv_d_dt_cdd) — the corner-div
    # contribution to the wind tendency.  We get this by taking
    # the difference between du_d_dt_on and du_d_dt_off, which is
    # entirely the corner-div d_con's wiring of the wind tendency
    # (it's gated INSIDE the same block).  But d_con DOES NOT
    # change du_d_dt — it only affects dT_dt.  So the wind
    # tendency is identical between cfg_on and cfg_off; we can
    # extract du_d_dt_cdd from EITHER.
    #
    # Strategy: compare against a "no corner-div" baseline.
    cfg_no_cdd = CDGridPrimitiveEquationConfig(
        **{k: v for k, v in common.items()
           if k != "corner_div_damp_d2_bg"},
        corner_div_damp_d2_bg=0.0,
        corner_div_damp_d_con=0.0,
    )
    tend_no_cdd = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_cdd, dt_actual=dt,
    )
    du_d_dt_cdd = tend_off.du_d_dt.data - tend_no_cdd.du_d_dt.data
    dv_d_dt_cdd = tend_off.dv_d_dt.data - tend_no_cdd.dv_d_dt.data

    u_d = state.u_d.data
    v_d = state.v_d.data
    dKE_dt_corner = u_d * du_d_dt_cdd + v_d * dv_d_dt_cdd
    dKE_dt_cc = interp_corner_to_center(dKE_dt_corner)
    expected_dT_dt = -1.0 * dKE_dt_cc / constants.c_pd

    np.testing.assert_allclose(
        dT_dt_d_con, expected_dT_dt,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "Corner-div d_con contribution to dT_dt does not "
            "match the expected formula "
            "-d_con * (u_d * du_dt_cdd + v_d * dv_dt_cdd) / c_pd "
            "projected to cell centres."
        ),
    )
