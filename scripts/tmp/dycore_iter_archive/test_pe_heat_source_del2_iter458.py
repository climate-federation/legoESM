"""FV3_3D iter 458: PE mirror of NH iter-457 del-2 smoothing
of aggregate heat_source.

Port of FV3 ``dyn_core.F90:1755-1756`` for the PE
``_d_con_sum`` aggregate (sum of corner_div, div_damp, A_h
slow-tendency d_con contributions).

Tests
-----

1. ``test_default_iters_is_zero``.
2. ``test_default_coeff_is_020``.
3. ``test_baseline_equals_no_smoothing``.
4. ``test_smoothing_changes_T``.
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
    rng = np.random.default_rng(seed=458)
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


def test_default_iters_is_zero():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.heat_source_del2_iters == 0


def test_default_coeff_is_020():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.heat_source_del2_coeff == 0.20


def test_baseline_equals_no_smoothing():
    grid, coord, state = _build_c8()
    common = dict(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    cfg_default = CDGridPrimitiveEquationConfig(**common)
    cfg_zero = CDGridPrimitiveEquationConfig(
        heat_source_del2_iters=0, **common,
    )
    m_d = CDGridPrimitiveEquationModel(grid, coord, cfg_default)
    m_z = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)
    s_d = m_d.step(state, dt=10.0)
    s_z = m_z.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.T.data), np.asarray(s_z.T.data),
        rtol=1e-14, atol=1e-14,
    )


def test_smoothing_changes_T():
    grid, coord, state = _build_c8()
    common = dict(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    cfg_off = CDGridPrimitiveEquationConfig(**common)
    cfg_on = CDGridPrimitiveEquationConfig(
        heat_source_del2_iters=2,
        heat_source_del2_coeff=0.20, **common,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.T.data - s_on.T.data)
    assert float(jnp.max(diff)) > 0.0, (
        "PE iter-458 del-2 smoothing produced no observable "
        "change in T."
    )
