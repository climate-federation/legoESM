"""FV3_3D iter 229: bit-for-bit formula tests for the iter-223
PE cell-centre div_damp d_con and the iter-225 PE A_h d_con
(continues iter-228 pattern for the two remaining PE d_con
sites).

Strategy: 3-config triangulation per site (matches iter-228).

Tests
-----

1. ``test_pe_div_damp_d_con_dT_dt_matches_formula`` — iter-223
   cell-centre div_damp d_con contribution to dT_dt matches
   ``-d_con * project_cc(u_d * du_d_dt_dd + v_d * dv_d_dt_dd) /
   c_pd`` at machine precision.
2. ``test_pe_ah_d_con_dT_dt_matches_formula`` — iter-225
   Smagorinsky-A_h d_con contribution to dT_dt matches
   ``-d_con * project_cc(u_d * du_d_dt_ah + v_d * dv_d_dt_ah) /
   c_pd`` at machine precision.
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
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.operators_cdgrid import interp_corner_to_center
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_pe_state_with_winds():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=229)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def test_pe_div_damp_d_con_dT_dt_matches_formula(small_pe_state_with_winds):
    """iter-223 PE cell-centre div_damp d_con contribution to
    dT_dt matches the expected formula at machine precision."""
    grid, cdgrid, coord, state = small_pe_state_with_winds

    common = dict(
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_dd = CDGridPrimitiveEquationConfig(
        **common, div_damp_coeff=0.0, div_damp_d_con=0.0,
    )
    cfg_dd_off = CDGridPrimitiveEquationConfig(
        **common, div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=0.0,
    )
    cfg_dd_on = CDGridPrimitiveEquationConfig(
        **common, div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
    )

    dt = 100.0
    tend_no_dd = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_dd, dt_actual=dt,
    )
    tend_dd_off = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_dd_off, dt_actual=dt,
    )
    tend_dd_on = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_dd_on, dt_actual=dt,
    )

    # Extract div_damp's contribution to (du_d_dt, dv_d_dt).
    du_d_dt_dd = tend_dd_off.du_d_dt.data - tend_no_dd.du_d_dt.data
    dv_d_dt_dd = tend_dd_off.dv_d_dt.data - tend_no_dd.dv_d_dt.data

    # Observed d_con contribution to dT_dt (cell centres).
    dT_dt_d_con = tend_dd_on.dT_dt.data - tend_dd_off.dT_dt.data

    # Expected formula.
    u_d = state.u_d.data
    v_d = state.v_d.data
    dKE_dt_corner = u_d * du_d_dt_dd + v_d * dv_d_dt_dd
    dKE_dt_cc = interp_corner_to_center(dKE_dt_corner)
    expected_dT_dt = -1.0 * dKE_dt_cc / constants.c_pd

    np.testing.assert_allclose(
        dT_dt_d_con, expected_dT_dt,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "PE cell-centre div_damp d_con contribution to dT_dt "
            "does not match the expected formula."
        ),
    )


def test_pe_ah_d_con_dT_dt_matches_formula(small_pe_state_with_winds):
    """iter-225 PE Smagorinsky-A_h d_con contribution to dT_dt
    matches the expected formula at machine precision."""
    grid, cdgrid, coord, state = small_pe_state_with_winds

    common = dict(
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_ah = CDGridPrimitiveEquationConfig(
        **common, A_h=0.0, smagorinsky_cs=0.0, ah_d_con=0.0,
    )
    cfg_ah_off = CDGridPrimitiveEquationConfig(
        **common, A_h=1e6, smagorinsky_cs=0.20, ah_d_con=0.0,
    )
    cfg_ah_on = CDGridPrimitiveEquationConfig(
        **common, A_h=1e6, smagorinsky_cs=0.20, ah_d_con=1.0,
    )

    dt = 100.0
    tend_no_ah = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_ah, dt_actual=dt,
    )
    tend_ah_off = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_ah_off, dt_actual=dt,
    )
    tend_ah_on = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_ah_on, dt_actual=dt,
    )

    # Extract A_h's contribution to (du_d_dt, dv_d_dt).
    du_d_dt_ah = tend_ah_off.du_d_dt.data - tend_no_ah.du_d_dt.data
    dv_d_dt_ah = tend_ah_off.dv_d_dt.data - tend_no_ah.dv_d_dt.data

    # Observed d_con contribution to dT_dt.
    dT_dt_d_con = tend_ah_on.dT_dt.data - tend_ah_off.dT_dt.data

    # Expected formula (note: A_h also contributes a Laplacian to
    # T directly via the lap_uvT[..., 2] term; that contribution
    # is the same in cfg_ah_off and cfg_ah_on so it cancels in
    # the subtraction).
    u_d = state.u_d.data
    v_d = state.v_d.data
    dKE_dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
    dKE_dt_cc = interp_corner_to_center(dKE_dt_corner)
    expected_dT_dt = -1.0 * dKE_dt_cc / constants.c_pd

    np.testing.assert_allclose(
        dT_dt_d_con, expected_dT_dt,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "PE Smagorinsky-A_h d_con contribution to dT_dt does "
            "not match the expected formula."
        ),
    )
