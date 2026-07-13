"""FV3_3D iter 377: C16 PE full FV3-fidelity stack with
iter-370 cross_face flag.  PE counterpart of iter-376.

Tests
-----

1. ``test_pe_c16_full_3_flags_does_not_amplify``
2. ``test_pe_c16_full_3_flags_changes_state``
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


def _build_c16(use_duogrid):
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=377)
    u_p = rng.uniform(-5.0, 5.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-5.0, 5.0, size=(6, n + 1, n + 1, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def _imprint_T(state, n_cells, edge_width=3):
    T = state.T.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    e = float(jnp.std(T[:, edge_mask, :].reshape(-1)))
    i = float(jnp.std(T[:, interior_mask, :].reshape(-1)))
    return e / max(i, 1e-30)


def _pe_cfg(all_flags):
    return CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        use_fv3_a2b_zeta_corner=all_flags,
        div_damp_coeff=1e6, div_damp_dddmp=0.20, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        delt_max=1.0,
        use_fv3_metric_aware_d_con=all_flags,
        use_fv3_cross_face_du_proj=all_flags,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )


def test_pe_c16_full_3_flags_does_not_amplify():
    grid_d, coord, state = _build_c16(False)
    grid_f, _, state_f = _build_c16(True)
    m_d = CDGridPrimitiveEquationModel(grid_d, coord, _pe_cfg(False))
    m_f = CDGridPrimitiveEquationModel(grid_f, coord, _pe_cfg(True))
    s_d = m_d.step(state, 100.0)
    s_f = m_f.step(state_f, 100.0)
    r_d = _imprint_T(s_d, n_cells=16)
    r_f = _imprint_T(s_f, n_cells=16)
    assert r_f <= r_d * 1.10


def test_pe_c16_full_3_flags_changes_state():
    grid_d, coord, state = _build_c16(False)
    grid_f, _, state_f = _build_c16(True)
    m_d = CDGridPrimitiveEquationModel(grid_d, coord, _pe_cfg(False))
    m_f = CDGridPrimitiveEquationModel(grid_f, coord, _pe_cfg(True))
    s_d = m_d.step(state, 100.0)
    s_f = m_f.step(state_f, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_d.T.data) - np.asarray(s_f.T.data),
    )))
    assert diff > 1e-6
