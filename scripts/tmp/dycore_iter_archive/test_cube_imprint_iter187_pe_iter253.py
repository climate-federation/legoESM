"""FV3_3D iter 253: PE counterpart of iter-252 cube-imprint
regression for the iter-187 smag_vort cap + iter-190 dedup
path.

iter-252 verified NH cube-imprint stays bounded with iter-187
ON.  iter-217 PE cube-imprint uses iter-190
(``use_fv3_a2b_zeta_corner=True``) but NOT iter-187
(``corner_div_damp_d4_bg=0`` in iter-217 cfg_toolkit).
iter 253 closes the PE gap.

Tests
-----

1. ``test_pe_iter187_smag_vort_does_not_amplify_imprint`` —
   iter-19 PE toolkit + iter-187 (nord=1, d4_bg > 0) +
   iter-190 (use_fv3_a2b_zeta_corner=True) ON; v_d imprint
   ratio at C8 stays within 50 % of the baseline
   (iter-19 toolkit without iter-187 d4_bg).
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


def _edge_interior_std_ratio_pe(state):
    """PE cube-imprint metric (iter-217 definition)."""
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


def test_pe_iter187_smag_vort_does_not_amplify_imprint():
    """PE iter-187 + iter-190 must not amplify cube-imprint
    ratio beyond 50 % of baseline (iter-168 nord=0)."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=253)
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
        corner_div_damp_d2_bg=0.001, corner_div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=0,
        div_damp_coeff=1e7,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_baseline = CDGridPrimitiveEquationConfig(
        **common,
        # Baseline: iter-168 nord=0, no iter-187 cap
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
        use_fv3_a2b_zeta_corner=False,
    )
    cfg_iter187 = CDGridPrimitiveEquationConfig(
        **common,
        # iter-187 path: nord=1 + d4_bg > 0 activates the
        # |dt|*sqrt(delpc²+ζ²) cap formula
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        # iter-190 dedup path: a2b zeta corner shared between
        # iter-170 site and iter-187 site
        use_fv3_a2b_zeta_corner=True,
    )

    def step5(cfg):
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        s = state
        for _ in range(5):
            s = m.step(s, 100.0)
        return s

    s_base = step5(cfg_baseline)
    s_iter187 = step5(cfg_iter187)

    ratio_base = _edge_interior_std_ratio_pe(s_base)
    ratio_iter187 = _edge_interior_std_ratio_pe(s_iter187)

    assert jnp.isfinite(ratio_base) and jnp.isfinite(ratio_iter187)
    assert 0.5 < ratio_base < 5.0
    assert 0.5 < ratio_iter187 < 5.0

    rel = abs(ratio_iter187 - ratio_base) / max(ratio_base, 1e-30)
    assert rel < 0.5, (
        f"PE iter-187 smag_vort + iter-190 dedup must not change "
        f"cube-imprint ratio by >50%: baseline={ratio_base:.4f}, "
        f"iter187={ratio_iter187:.4f}, rel={rel:.3f}.  Possible "
        f"PE-side edge artifact from the iter-187 cap or iter-190 "
        f"dedup path."
    )
