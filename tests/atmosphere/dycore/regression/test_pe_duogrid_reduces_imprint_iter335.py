"""FV3_3D iter 335: quantitative cube-imprint impact from iter-333
PE duogrid wiring at the FV3 corner-div damping ke_correction
halo.  Mirrors iter-326 (NH version).

iter-333 verified the wiring CHANGES PE wind tendency when
``use_duogrid=True`` (proves wiring active).  iter-335 closes the
directional impact gap: with corner-div damping engaged, the
duogrid path's wind diff vs no-duogrid concentrates at panel
edges (cube boundary cells) — proves the halo correction lands at
the cube-edge halo cells, not a global shift no-op.

Tests
-----

1. ``test_pe_duogrid_diff_concentrated_at_edges`` — wind-tendency
   diff between duogrid + no-duogrid PE paths concentrates at
   panel edges (edge_max > 1.5 × interior_max for the corner-
   stagger u_d field).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build(use_duogrid: bool):
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=335)
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


def test_pe_duogrid_diff_concentrated_at_edges():
    """PE duogrid - no_duogrid wind-tendency diff concentrates at
    panel edges.  Proves iter-333 wiring lands at cube-boundary
    halo cells (not bulk shift no-op)."""
    grid_plain, cdgrid_plain, coord, state_plain = _build(False)
    grid_duo, cdgrid_duo, _, state_duo = _build(True)

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    tend_plain = fv3_hydrostatic_tendencies(
        state_plain, grid_plain, coord, cdgrid_plain, cfg,
        dt_actual=100.0,
    )
    tend_duo = fv3_hydrostatic_tendencies(
        state_duo, grid_duo, coord, cdgrid_duo, cfg,
        dt_actual=100.0,
    )

    diff = (
        np.asarray(tend_duo.du_d_dt.data)
        - np.asarray(tend_plain.du_d_dt.data)
    )
    # u_d at corners shape (6, n+1, n+1, nlev).
    n_c = grid_plain.n + 1
    edge_width = 2
    i_idx = np.arange(n_c)
    edge_i = (i_idx < edge_width) | (i_idx >= n_c - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask

    edge_max = float(np.max(np.abs(diff[:, edge_mask, :])))
    interior_max = float(np.max(np.abs(diff[:, interior_mask, :])))

    assert edge_max > interior_max * 1.5, (
        f"PE duogrid - no_duogrid diff is NOT concentrated at "
        f"panel edges: edge_max={edge_max:.3e} ≤ 1.5 * "
        f"interior_max={1.5 * interior_max:.3e}.  iter-333 wiring "
        f"may have a propagation bug (silent interior effect) "
        f"or be a global shift no-op."
    )
