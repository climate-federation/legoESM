"""FV3_3D iter 433: PE mirror of NH iter-431/432 — FV3 sponge
zeroing of d_con KE→heat in top N levels.

PE has 4 d_con sites (vs NH's 5):
* post-acoustic damp_v (iter-208) — mirror of NH iter-431
* aggregate slow-tendency sum covering 3 sites (iter-221/223/
  225: corner_div, div_damp, A_h) — mirror of NH iter-432

Config field ``d_con_top_zero_levels: int = 0`` added to
``CDGridPrimitiveEquationConfig``.  Default 0 preserves
bit-for-bit baseline.

Tests
-----

1. ``test_default_value_is_zero`` — config field defaults to 0.
2. ``test_damp_v_site_masked`` — damp_v d_con site respects
   mask: top-level T change concentrated; bottom levels
   identical at FP precision when w forcing absent.
3. ``test_slow_tendency_aggregate_masked`` — slow-tendency
   d_con aggregate respects mask: top-level T change >>
   bottom drift.
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


def _build_c8():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=433)
    n_corners = n + 1
    u_p = rng.uniform(-3.0, 3.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-3.0, 3.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def test_default_value_is_zero():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.d_con_top_zero_levels == 0


def test_damp_v_site_masked():
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        d_con_top_zero_levels=0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        d_con_top_zero_levels=2,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    dT_off = s_off.T.data - state.T.data
    dT_on = s_on.T.data - state.T.data
    diff_top = float(jnp.max(jnp.abs(
        dT_off[..., :2] - dT_on[..., :2],
    )))
    diff_bot = float(jnp.max(jnp.abs(
        dT_off[..., 2:] - dT_on[..., 2:],
    )))
    assert diff_top > 0.0, "PE damp_v d_con mask is a no-op."
    assert diff_top > 10.0 * diff_bot, (
        f"PE damp_v mask did not concentrate in top: "
        f"top {diff_top:.2e}, bot {diff_bot:.2e}."
    )


def test_slow_tendency_aggregate_masked():
    grid, coord, state = _build_c8()
    common = dict(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    cfg_off = CDGridPrimitiveEquationConfig(
        d_con_top_zero_levels=0, **common,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        d_con_top_zero_levels=2, **common,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    dT_off = s_off.T.data - state.T.data
    dT_on = s_on.T.data - state.T.data
    diff_top = float(jnp.max(jnp.abs(
        dT_off[..., :2] - dT_on[..., :2],
    )))
    diff_bot = float(jnp.max(jnp.abs(
        dT_off[..., 2:] - dT_on[..., 2:],
    )))
    assert diff_top > 0.0, "PE slow-tendency aggregate mask is a no-op."
    assert diff_top > 10.0 * diff_bot, (
        f"PE slow-tendency mask did not concentrate in top: "
        f"top {diff_top:.2e}, bot {diff_bot:.2e}."
    )
