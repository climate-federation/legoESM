"""FV3_3D iter 233: cube-imprint regression for the full PE
d_con stack.

iter-217 verified that the iter-19 PE toolkit measurably changes
the edge_std/interior_std ratio for v_d under random IC at C8
(no strong amplification).  iter 233 extends that check: with
the full PE d_con stack ON (corner_div_damp_d_con,
div_damp_d_con, ah_d_con, damp_v_d_con all = 1.0), the imprint
ratio must remain in the same sanity bound — d_con should not
introduce edge artifacts because the d_con heat is added to
``dT_dt`` at cell centres (no edge stencil).

Tests
-----

1. ``test_pe_d_con_does_not_amplify_imprint_ratio`` — toolkit
   ON without d_con and toolkit ON with d_con stack at 1.0 each
   produce imprint ratios within 50 % of each other (catches
   any d_con-induced edge artifact).
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


def _edge_interior_std_ratio_pe(state):
    """Cube-imprint metric for PE — ratio of v_d edge cell std to
    interior cell std.  Reuses iter-217's metric definition."""
    v = state.v_d.data    # (6, n+1, n+1, nlev)
    n = v.shape[1] - 1
    edge_width = 2

    i_idx = jnp.arange(n + 1)
    j_idx = jnp.arange(n + 1)
    edge_i = (i_idx < edge_width) | (i_idx >= (n + 1) - edge_width)
    edge_j = (j_idx < edge_width) | (j_idx >= (n + 1) - edge_width)
    edge_mask = edge_i[:, None] | edge_j[None, :]
    interior_mask = ~edge_mask

    v_edge = v[:, edge_mask, :].reshape(-1)
    v_interior = v[:, interior_mask, :].reshape(-1)

    return float(jnp.std(v_edge)) / max(float(jnp.std(v_interior)), 1e-30)


def test_pe_d_con_does_not_amplify_imprint_ratio():
    """Adding the full PE d_con stack on top of the iter-19
    toolkit must not strongly amplify the cube-imprint ratio."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=233)
    n_corners = n + 1
    u_p = rng.uniform(-1.0, 1.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-1.0, 1.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    common = dict(
        # iter-19 PE toolkit (matches iter-217 setup) + full
        # damping stack so all d_con sites engage.
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

    def step5(cfg):
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        s = state
        for _ in range(5):
            s = m.step(s, 100.0)
        return s

    s_off = step5(cfg_off)
    s_on = step5(cfg_on)

    ratio_off = _edge_interior_std_ratio_pe(s_off)
    ratio_on = _edge_interior_std_ratio_pe(s_on)

    # Both must be finite and within sanity bounds.
    assert jnp.isfinite(ratio_off) and jnp.isfinite(ratio_on)
    assert 0.5 < ratio_off < 5.0
    assert 0.5 < ratio_on < 5.0

    # d_con ON should not amplify the ratio more than 50 %
    # relative to d_con OFF.  d_con only modifies T tendency at
    # cell centres (no edge stencil) so the v_d field structure
    # at edges should be essentially unchanged from d_con's
    # direct effect; any large amplification would indicate
    # spurious edge response (e.g., through a feedback loop the
    # d_con T change drives via PGF).
    rel = abs(ratio_on - ratio_off) / max(ratio_off, 1e-30)
    assert rel < 0.5, (
        f"PE d_con stack must not change cube-imprint ratio by "
        f">50%: ratio_off={ratio_off:.4f}, "
        f"ratio_on={ratio_on:.4f}, rel={rel:.3f}.  Possible "
        f"d_con-induced edge artifact."
    )
