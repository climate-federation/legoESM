"""FV3_3D iter 386: cross_face flag at C16 WITH duogrid grid.

iter-372 used default (non-duogrid) grid where cross_face is a
no-op per iter-384 finding.  iter-386 re-runs at C16 with
``use_duogrid=True`` so the flag has actual effect.

Tests
-----

1. ``test_pe_c16_duogrid_cross_face_changes_state`` — at C16
   duogrid, cross_face=True measurably differs from
   cross_face=False.
2. ``test_pe_c16_duogrid_cross_face_does_not_amplify`` — and
   does not amplify imprint by > 10 %.
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


def _build_c16_duogrid():
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=386)
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


def _cfg(cross):
    return CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
        corner_div_damp_d2_bg=0.0, A_h=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_cross_face_du_proj=cross,
    )


def test_pe_c16_duogrid_cross_face_changes_state():
    grid, coord, state = _build_c16_duogrid()
    m_off = CDGridPrimitiveEquationModel(grid, coord, _cfg(False))
    m_on = CDGridPrimitiveEquationModel(grid, coord, _cfg(True))
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.u_d.data) - np.asarray(s_off.u_d.data),
    )))
    assert diff > 1e-8, (
        f"cross_face=True with duogrid did not change u_d "
        f"measurably (diff={diff:.3e})."
    )


def test_pe_c16_duogrid_cross_face_does_not_amplify():
    grid, coord, state = _build_c16_duogrid()
    m_off = CDGridPrimitiveEquationModel(grid, coord, _cfg(False))
    m_on = CDGridPrimitiveEquationModel(grid, coord, _cfg(True))
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    r_off = _imprint_T(s_off, n_cells=16)
    r_on = _imprint_T(s_on, n_cells=16)
    assert r_on <= r_off * 1.10
