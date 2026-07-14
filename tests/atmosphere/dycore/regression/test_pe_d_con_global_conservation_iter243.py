"""FV3_3D iter 243: global-energy-conservation regression for
all 3 PE tendency-based d_con sites (extends iter-238).

iter-238 verified ``Σ c_pd * dT_dt + Σ dKE_dt == 0`` for the
corner-div d_con block alone.  iter-243 extends to all 3 PE
tendency d_con sites individually:

* iter-221 corner_div_damp_d_con
* iter-223 div_damp_d_con
* iter-225 ah_d_con

Plus the iter-239 aggregate path (all 3 simultaneously).

The conservation property must hold for EACH site and for
the aggregate at machine precision (rtol=1e-10), since each
formula is ``-d_con * dKE_dt_cc / c_pd`` and the aggregate is
just the linear sum.

Tests
-----

1. ``test_corner_div_damp_d_con_conserves`` (iter-221).
2. ``test_div_damp_d_con_conserves`` (iter-223).
3. ``test_ah_d_con_conserves`` (iter-225).
4. ``test_aggregate_d_con_conserves`` (iter-239 path with all
   3 ON simultaneously, no cap).
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

    rng = np.random.default_rng(seed=243)
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


def _check_conservation(state, grid, coord, cdgrid,
                        cfg_no_X, cfg_X_off, cfg_X_on, dt=100.0):
    """Compute c_pd * Σ dT_dt and -Σ dKE_dt for the d_con
    contribution of mechanism X, verify they balance at rtol=1e-10."""
    tend_no_X = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_X, dt_actual=dt,
    )
    tend_X_off = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_X_off, dt_actual=dt,
    )
    tend_X_on = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_X_on, dt_actual=dt,
    )

    # X's contribution to wind tendency.
    du_d_dt_X = tend_X_off.du_d_dt.data - tend_no_X.du_d_dt.data
    dv_d_dt_X = tend_X_off.dv_d_dt.data - tend_no_X.dv_d_dt.data

    # X's d_con contribution to T tendency.
    dT_dt_d_con = tend_X_on.dT_dt.data - tend_X_off.dT_dt.data

    # Compute KE rate at corners then project to cell centres.
    u_d = state.u_d.data
    v_d = state.v_d.data
    dKE_dt_corner = u_d * du_d_dt_X + v_d * dv_d_dt_X
    dKE_dt_cc = interp_corner_to_center(dKE_dt_corner)

    # Conservation: Σ c_pd * dT_dt = -Σ dKE_dt
    total_heat = float(jnp.sum(constants.c_pd * dT_dt_d_con))
    total_KE_loss = float(jnp.sum(dKE_dt_cc))
    np.testing.assert_allclose(
        total_heat, -total_KE_loss,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "PE d_con violates global energy conservation:"
            f" Σ c_pd*dT/dt={total_heat:.4e}, "
            f"Σ dKE/dt={total_KE_loss:.4e}, "
            f"residual={total_heat + total_KE_loss:.4e}."
        ),
    )

    # Sanity: non-vacuous.
    assert abs(total_heat) > 1e-3


def test_corner_div_damp_d_con_conserves(small_pe_state_with_winds):
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common_off = dict(
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_X = CDGridPrimitiveEquationConfig(
        **common_off,
        corner_div_damp_d2_bg=0.0,
        corner_div_damp_d_con=0.0,
    )
    cfg_X_off = CDGridPrimitiveEquationConfig(
        **common_off,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=0.0,
    )
    cfg_X_on = CDGridPrimitiveEquationConfig(
        **common_off,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
    )
    _check_conservation(state, grid, coord, cdgrid,
                        cfg_no_X, cfg_X_off, cfg_X_on)


def test_div_damp_d_con_conserves(small_pe_state_with_winds):
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common_off = dict(
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_X = CDGridPrimitiveEquationConfig(
        **common_off,
        div_damp_coeff=0.0,
        div_damp_d_con=0.0,
    )
    cfg_X_off = CDGridPrimitiveEquationConfig(
        **common_off,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=0.0,
    )
    cfg_X_on = CDGridPrimitiveEquationConfig(
        **common_off,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
    )
    _check_conservation(state, grid, coord, cdgrid,
                        cfg_no_X, cfg_X_off, cfg_X_on)


def test_ah_d_con_conserves(small_pe_state_with_winds):
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common_off = dict(
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_X = CDGridPrimitiveEquationConfig(
        **common_off,
        A_h=0.0, smagorinsky_cs=0.0,
        ah_d_con=0.0,
    )
    cfg_X_off = CDGridPrimitiveEquationConfig(
        **common_off,
        A_h=1e6, smagorinsky_cs=0.20,
        ah_d_con=0.0,
    )
    cfg_X_on = CDGridPrimitiveEquationConfig(
        **common_off,
        A_h=1e6, smagorinsky_cs=0.20,
        ah_d_con=1.0,
    )
    _check_conservation(state, grid, coord, cdgrid,
                        cfg_no_X, cfg_X_off, cfg_X_on)


def test_aggregate_d_con_conserves(small_pe_state_with_winds):
    """All 3 d_con sites ON simultaneously (iter-239 aggregate
    path); verify global conservation holds for the sum."""
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common = dict(
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        use_conservation_fixer=False, fix_mass=False,
    )
    # All damping ON, no d_con
    cfg_off = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0, ah_d_con=0.0,
    )
    # All damping ON, all 3 d_con knobs ON, no cap
    cfg_on = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=0.0,    # no cap so conservation holds at machine precision
    )
    # Reference with all damping OFF for du_d_dt extraction.
    cfg_no_damp = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        corner_div_damp_d2_bg=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, smagorinsky_cs=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )

    dt = 100.0
    tend_no = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_damp, dt_actual=dt,
    )
    tend_off = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_off, dt_actual=dt,
    )
    tend_on = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_on, dt_actual=dt,
    )

    # Combined wind tendency from all 3 damping mechanisms.
    du_d_dt_all = tend_off.du_d_dt.data - tend_no.du_d_dt.data
    dv_d_dt_all = tend_off.dv_d_dt.data - tend_no.dv_d_dt.data

    # Aggregate d_con contribution to T tendency.
    dT_dt_d_con = tend_on.dT_dt.data - tend_off.dT_dt.data

    u_d = state.u_d.data
    v_d = state.v_d.data
    dKE_dt_corner = u_d * du_d_dt_all + v_d * dv_d_dt_all
    dKE_dt_cc = interp_corner_to_center(dKE_dt_corner)

    total_heat = float(jnp.sum(constants.c_pd * dT_dt_d_con))
    total_KE_loss = float(jnp.sum(dKE_dt_cc))
    np.testing.assert_allclose(
        total_heat, -total_KE_loss,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "Aggregate PE d_con violates global energy conservation: "
            f"Σ c_pd*dT/dt={total_heat:.4e}, "
            f"Σ dKE/dt={total_KE_loss:.4e}, "
            f"residual={total_heat + total_KE_loss:.4e}."
        ),
    )
