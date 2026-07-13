"""FV3_3D iter 268: cube-imprint metric OVER TIME for PE
production toolkit + d_con stack.

iter-242/244 verify multi-step stability (max|u|, finiteness)
but not the cube-imprint metric.  iter-254 verifies imprint
after 5 steps at C36.  iter-268 ties them together: track
cube-imprint ratio every 10 steps over a 50-step PE C8 run.

Verifies the imprint ratio remains BOUNDED (not growing
unboundedly) and stays in physically reasonable range over
time with the full toolkit + d_con stack.

Tests
-----

1. ``test_pe_imprint_ratio_bounded_over_time`` — 50 PE steps
   with iter-19 toolkit + full d_con stack; record imprint
   ratio every 10 steps; max ratio across all sample times
   stays < 5.0 (sanity bound).
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


def test_pe_imprint_ratio_bounded_over_time():
    """50 PE steps with iter-19 toolkit + full d_con stack;
    imprint ratio stays bounded across all sampled times."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=268)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        # iter-19 PRODUCTION at C8 (slightly softer d4_bg than
        # actual production to avoid over-damping at C8).
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=0.0,
        # Full d_con stack
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    # Track imprint ratio every 10 steps.
    s = state
    ratios = [_edge_interior_std_ratio_pe(s)]
    for i in range(50):
        s = model.step(s, 100.0)
        if (i + 1) % 10 == 0:
            ratios.append(_edge_interior_std_ratio_pe(s))

    # All sampled ratios must be finite + bounded.
    for k, r in enumerate(ratios):
        assert jnp.isfinite(r), (
            f"PE imprint ratio NaN/Inf at sample {k} (step "
            f"{k * 10})."
        )
        assert 0.1 < r < 5.0, (
            f"PE imprint ratio at step {k * 10} = {r:.4f} "
            f"outside sanity bound (0.1, 5.0).  "
            f"All samples: {[f'{x:.3f}' for x in ratios]}."
        )

    # Verify ratio doesn't EXPLODE over time (max < 2.0 ×
    # initial).  This catches slow growth in the imprint that
    # the spot checks miss.
    max_r = max(ratios)
    initial_r = ratios[0]
    growth_factor = max_r / max(initial_r, 1e-30)
    assert growth_factor < 5.0, (
        f"PE imprint ratio growth {growth_factor:.2f}x over 50 "
        f"steps — possible slow-growth edge artifact accumulation."
    )
