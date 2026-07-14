"""FV3_3D iter 256: global-energy-conservation regression for
the iter-208 damp_v_d_con post-step block (PE).

iter-243 covered all 3 PE TENDENCY-based d_con sites
(corner-div, cell-centre div_damp, A_h).  iter-208 damp_v_d_con
is structurally different — a POST-STEP discrete block — so it
wasn't included in iter-243.  iter-256 closes that gap.

Conservation property (post-step discrete form):

    c_pd * ΔT_d_con_cc = -project_cc(ΔKE_corner)
    where ΔKE_corner = u_d * du + 0.5*du² + v_d * dv + 0.5*dv²

This is the EXACT discrete energy budget — the 0.5*du² nonlinear
terms ARE included (unlike the tendency-form drop in iter-208
docstring).

Tests
-----

1. ``test_pe_damp_v_d_con_post_step_conserves_global_energy`` —
   strong-wind IC, single PE step, verify
   ``Σ c_pd * ΔT_d_con_cc + Σ project_cc(ΔKE_corner) == 0`` at
   rtol=1e-10.
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
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.operators_cdgrid import interp_corner_to_center
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_damp_v_d_con_post_step_conserves_global_energy():
    """PE iter-208 damp_v_d_con post-step block conserves
    global energy: Σ c_pd * ΔT_d_con + Σ ΔKE = 0."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=256)
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
        # Only damp_v active so we isolate the iter-208 post-step
        # contribution.  No corner-div, no div_damp, no A_h.
        damp_v=0.030, nord_v=2,
        corner_div_damp_d2_bg=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_dcon = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=0.0,
    )
    cfg_dcon_on = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=1.0,
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no_dcon)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_dcon_on)

    s_no = m_no.step(state, 100.0)
    s_on = m_on.step(state, 100.0)

    # damp_v's contribution to (u_d, v_d) is THEORETICALLY the
    # same in both runs since d_con only affects T tendency.
    # In practice tiny FP-roundoff differences are possible
    # (e.g., a JIT optimization order change between cfg_no_dcon
    # and cfg_dcon_on JIT specializations).  Verify within 1
    # ULP.
    np.testing.assert_allclose(
        s_no.u_d.data, s_on.u_d.data,
        rtol=1e-14, atol=1e-15,
        err_msg="damp_v wind change must be identical (modulo "
                "FP roundoff) between d_con=0 and d_con=1.",
    )
    np.testing.assert_allclose(
        s_no.v_d.data, s_on.v_d.data, rtol=1e-14, atol=1e-15,
    )

    # damp_v's contribution to T_d_con: difference between
    # d_con=1 and d_con=0 runs.
    dT_d_con = s_on.T.data - s_no.T.data

    # Compute expected ΔKE per unit mass at corners using the
    # iter-208 EXACT discrete formula.
    # The wind change `du_corner = u_d_new - u_d_old` includes
    # both advection AND the damp_v post-step.  Since the
    # advection contribution is the same in both runs (cfg_no
    # and cfg_on), and damp_v also same, the per-step change is
    # identical between runs and represents the FULL change.
    # For conservation we need the damp_v ONLY contribution.
    #
    # Strategy: also run with damp_v=0 → no wind change from
    # damp_v post-step (only advection-induced change).  Then
    # the difference gives damp_v's contribution.
    cfg_no_damp_v = CDGridPrimitiveEquationConfig(
        **{k: v for k, v in common.items() if k != "damp_v"},
        damp_v=0.0,
    )
    m_no_damp = CDGridPrimitiveEquationModel(
        grid, coord, cfg_no_damp_v,
    )
    s_no_damp = m_no_damp.step(state, 100.0)

    # damp_v wind contribution = (with damp_v) - (without damp_v).
    du_damp_v = s_on.u_d.data - s_no_damp.u_d.data
    dv_damp_v = s_on.v_d.data - s_no_damp.v_d.data

    # ΔKE per unit mass at corners — iter-208 formula.
    u_d_initial = state.u_d.data
    v_d_initial = state.v_d.data
    # u_d at the time the damp_v post-step ACTS is u_d_no_damp
    # (after advection but before damp_v).
    u_d_pre_damp = s_no_damp.u_d.data
    v_d_pre_damp = s_no_damp.v_d.data
    dKE_corner = (
        u_d_pre_damp * du_damp_v + 0.5 * du_damp_v ** 2
        + v_d_pre_damp * dv_damp_v + 0.5 * dv_damp_v ** 2
    )
    dKE_cc = interp_corner_to_center(dKE_corner)

    # damp_v's contribution to T_d_con (already extracted as
    # dT_d_con = s_on - s_no above; in the no-cfg run dT_d_con
    # has been zeroed because damp_v_d_con=0).
    expected_dT_d_con_cc = -1.0 * dKE_cc / constants.c_pd

    # Per-cell pointwise check (similar to iter-208 internal
    # logic).
    np.testing.assert_allclose(
        dT_d_con, expected_dT_d_con_cc,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "iter-208 damp_v_d_con per-cell formula must match "
            "the EXACT discrete energy budget (with 0.5*du² and "
            "0.5*dv² nonlinear terms)."
        ),
    )

    # Global conservation: Σ c_pd * ΔT_d_con + Σ project(ΔKE) = 0.
    total_heat = float(jnp.sum(constants.c_pd * dT_d_con))
    total_KE = float(jnp.sum(dKE_cc))
    np.testing.assert_allclose(
        total_heat, -total_KE,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "iter-208 damp_v_d_con violates global energy "
            "conservation."
        ),
    )

    # Sanity: non-vacuous.
    assert abs(total_heat) > 1e-3
