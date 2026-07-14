"""FV3_3D iter 231: aggregate direction test for the full PE
d_con stack.

iter-228 / iter-229 verified that EACH PE d_con site
(corner-div, cell-centre div_damp, A_h) has the correct
formula in tendency form.  iter 231 verifies that the
AGGREGATE behaves correctly under integration: when all four
PE d_con knobs (damp_v_d_con, corner_div_damp_d_con,
div_damp_d_con, ah_d_con) are ON simultaneously and damping
mechanisms are removing KE, the column mean ``c_p * T`` should
be HIGHER than with d_con OFF (energy conservation: KE→heat).

Tests
-----

1. ``test_pe_full_d_con_stack_heats_column_vs_off`` — strong
   wind perturbation IC, 10 PE steps with all damping ON.
   ``mean(T)_d_con_ON > mean(T)_d_con_OFF``.
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


def test_pe_full_d_con_stack_heats_column_vs_off():
    """Full PE damping stack ON, with all 4 d_con knobs ON, must
    heat the column relative to the d_con-OFF baseline (energy
    conservation: KE→heat)."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=231)
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
        # Full PE damping stack ON (mirrors iter-185 umbrella).
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0,
        ah_d_con=0.0,
        damp_v_d_con=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        damp_v_d_con=1.0,
    )

    def run(cfg, n_steps=10, dt=100.0):
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        s = state
        for _ in range(n_steps):
            s = m.step(s, dt)
        return s

    s_off = run(cfg_off)
    s_on = run(cfg_on)

    mean_T_off = float(jnp.mean(s_off.T.data))
    mean_T_on = float(jnp.mean(s_on.T.data))

    # With d_con ON, the KE removed by damping mechanisms is
    # converted to heat; mean(T) should be measurably higher.
    delta_mean_T = mean_T_on - mean_T_off
    assert delta_mean_T > 0.0, (
        f"All-d_con-ON column must be WARMER than all-d_con-OFF: "
        f"mean(T)_ON={mean_T_on:.6f}, mean(T)_OFF={mean_T_off:.6f}, "
        f"Δ={delta_mean_T:.6e}.  Either an iter-208/221/223/225 "
        f"d_con site has its sign wrong or energy conservation "
        f"is broken in the aggregate."
    )

    # Sanity: the heat added should be physically reasonable
    # (not exploding).  Strong-wind IC + 10 steps should be
    # ≪ 1 K of mean heating; if delta is huge, something is amiss.
    assert delta_mean_T < 5.0, (
        f"Δmean(T)={delta_mean_T:.4e} K is unphysically large "
        f"for 10 PE steps with C8 random IC.  Possible runaway "
        f"in d_con aggregation."
    )
