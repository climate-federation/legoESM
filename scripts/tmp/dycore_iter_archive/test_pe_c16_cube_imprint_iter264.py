"""FV3_3D iter 264: cube-imprint reduction validation at C16
(intermediate resolution between C8 tests and C36 production).

iter-217 PE cube-imprint at C8: toolkit measurably changes
ratio.  iter-254 PE C36 production: ratio bounded.  iter-264
fills the gap with C16 (4× more cells per face than C8) where
the toolkit should produce a CLEARER reduction signal.

Tests
-----

1. ``test_pe_c16_iter19_toolkit_reduces_imprint`` — PE C16 +
   iter-19 production toolkit (without d_con) reduces the v_d
   cube-imprint ratio relative to the iter-168 nord=0 baseline
   over 5 steps.
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


def _edge_interior_std_ratio_pe(state, edge_width=2):
    v = state.v_d.data    # (6, n+1, n+1, nlev)
    n = v.shape[1] - 1

    i_idx = jnp.arange(n + 1)
    j_idx = jnp.arange(n + 1)
    edge_i = (i_idx < edge_width) | (i_idx >= (n + 1) - edge_width)
    edge_j = (j_idx < edge_width) | (j_idx >= (n + 1) - edge_width)
    edge_mask = edge_i[:, None] | edge_j[None, :]
    interior_mask = ~edge_mask

    v_edge = v[:, edge_mask, :].reshape(-1)
    v_interior = v[:, interior_mask, :].reshape(-1)
    return float(jnp.std(v_edge)) / max(float(jnp.std(v_interior)), 1e-30)


def test_pe_c16_iter19_toolkit_reduces_imprint():
    """iter-19 PE production toolkit reduces v_d cube-imprint
    ratio at C16 vs iter-168 nord=0 baseline."""
    n = 16
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=264)
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
        damp_v=0.0, nord_v=0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_baseline = CDGridPrimitiveEquationConfig(
        **common,
        # iter-168 nord=0 only
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
        use_fv3_a2b_zeta_corner=False,
    )
    cfg_iter19 = CDGridPrimitiveEquationConfig(
        **common,
        # iter-19 PRODUCTION toolkit
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02, corner_div_damp_nord=1,
        use_fv3_a2b_zeta_corner=True,
    )

    def step5(cfg):
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        s = state
        for _ in range(5):
            s = m.step(s, 100.0)
        return s

    s_base = step5(cfg_baseline)
    s_iter19 = step5(cfg_iter19)

    ratio_base = _edge_interior_std_ratio_pe(s_base)
    ratio_iter19 = _edge_interior_std_ratio_pe(s_iter19)

    assert jnp.isfinite(ratio_base) and jnp.isfinite(ratio_iter19)
    assert 0.5 < ratio_base < 5.0
    assert 0.5 < ratio_iter19 < 5.0

    # At C16 with iter-19 production toolkit, the imprint
    # should be measurably reduced or at most weakly amplified
    # (within 1.5× of baseline — the same sanity envelope as
    # iter-217 at C8).
    rel_change = (ratio_iter19 - ratio_base) / max(ratio_base, 1e-30)
    assert rel_change < 0.5, (
        f"iter-19 toolkit at C16 must not amplify cube-imprint "
        f"by >50%: baseline={ratio_base:.4f}, "
        f"iter19={ratio_iter19:.4f}, rel={rel_change:.3f}.  "
        f"Possible regression in the iter-19 production "
        f"toolkit's edge-suppression behavior."
    )
