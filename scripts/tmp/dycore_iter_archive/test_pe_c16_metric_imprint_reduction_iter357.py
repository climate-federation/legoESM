"""FV3_3D iter 357: empirical cube-imprint reduction at C16 with
PE metric-aware d_con flag ON vs default.

iter-264 C16 cube-imprint baseline with iter-19 toolkit (default
flags); iter-357 measures additional reduction from
``use_fv3_metric_aware_d_con=True``.

At C16 the cube-imprint signal is bigger than C8 (per iter-179
finding), so the metric correction should be measurable.

Tests
-----

1. ``test_pe_c16_metric_does_not_amplify`` — metric flag does
   NOT amplify cube-imprint ratio vs default at C16.  Bounded
   regression: metric is opt-in fidelity improvement, must not
   make production state WORSE.
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


def _build_c16():
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=357)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _imprint_ratio_T(state, n_cells, edge_width=3):
    """T-field edge-vs-interior std ratio (cell-centre)."""
    T = state.T.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge = T[:, edge_mask, :].reshape(-1)
    interior = T[:, interior_mask, :].reshape(-1)
    e = float(jnp.std(edge))
    i = float(jnp.std(interior))
    return e / max(i, 1e-30)


def _toolkit_cfg(metric):
    return CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1,
        damp_v_d_con=1.0,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        delt_max=1.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_metric_aware_d_con=metric,
    )


def test_pe_c16_metric_does_not_amplify():
    """Metric flag at C16 does NOT amplify cube-imprint ratio
    vs default (no metric).  Bounded regression: opt-in
    fidelity improvement must not make production state worse.
    """
    grid, _, coord, state = _build_c16()
    m_default = CDGridPrimitiveEquationModel(
        grid, coord, _toolkit_cfg(metric=False),
    )
    m_metric = CDGridPrimitiveEquationModel(
        grid, coord, _toolkit_cfg(metric=True),
    )
    s_d = m_default.step(state, 100.0)
    s_m = m_metric.step(state, 100.0)
    r_d = _imprint_ratio_T(s_d, n_cells=16)
    r_m = _imprint_ratio_T(s_m, n_cells=16)
    # Metric flag is opt-in cube-edge correction.  Must not amplify
    # imprint vs default at C16.  Tolerance: 1.10 (10 %).
    assert r_m <= r_d * 1.10, (
        f"PE metric flag at C16 AMPLIFIES imprint by >10%: "
        f"r_default={r_d:.4f}, r_metric={r_m:.4f}.  Either the "
        f"metric form has wrong sign or the cosa_s correction "
        f"destabilizes at C16."
    )
