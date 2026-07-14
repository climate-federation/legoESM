"""FV3_3D iter 443: PE mirror of NH iter-442 sponge boost of
``damp_v``.

Ports FV3 ``dyn_core.F90:786-787, 796-797`` ``damp_vt = 0.5 *
d2_divg`` at sponge layers k=0, k=1 (FV3 does NOT extend to
k=2 for damp_v).

Tests
-----

1. ``test_default_flag_false``.
2. ``test_flag_false_no_change``.
3. ``test_flag_on_no_d2_bg_no_change``.
4. ``test_flag_on_k1_boost_changes_u_d``.
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


def _build_c8():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=443)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def test_default_flag_false():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.use_fv3_sponge_damp_v is False


def test_flag_false_no_change():
    grid, coord, state = _build_c8()
    cfg_d = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
    )
    cfg_f = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_v=False,
    )
    m_d = CDGridPrimitiveEquationModel(grid, coord, cfg_d)
    m_f = CDGridPrimitiveEquationModel(grid, coord, cfg_f)
    s_d = m_d.step(state, dt=10.0)
    s_f = m_f.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.u_d.data), np.asarray(s_f.u_d.data),
        rtol=1e-14, atol=1e-14,
    )


def test_flag_on_no_d2_bg_no_change():
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        use_fv3_sponge_damp_v=False,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        use_fv3_sponge_damp_v=True,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_off.u_d.data), np.asarray(s_on.u_d.data),
        rtol=1e-14, atol=1e-14,
    )


def test_flag_on_k1_boost_changes_u_d():
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_v=False,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_v=True,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.u_d.data - s_on.u_d.data)
    diff_top = float(jnp.max(diff[..., 0]))
    assert diff_top > 0.0, (
        "iter-443 PE damp_v sponge boost is a no-op at k=0."
    )
