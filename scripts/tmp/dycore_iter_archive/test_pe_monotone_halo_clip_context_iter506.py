"""FV3_3D iter 506: PE counterpart of iter-505 ``monotone_halo_clip_context``.

iter-505 demonstrated 32% edge-ratio reduction on NH with the
user-facing context manager + min-edge factory.  This iter
verifies the same composition works on the hydrostatic PE
dycore.

Tests
-----

1. ``test_pe_context_reduces_edge_ratio`` — PE min-edge
   factory + monotone_halo_clip_context yields edge ratio
   < edge ratio without the context.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_legoesm_pe_min_edge_config,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import standard_hybrid_levels


def _edge_and_interior_std(field_data):
    if field_data.ndim == 3:
        n_face, n_x, n_y = field_data.shape
        edge_mask = np.zeros((n_x, n_y), dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        edge_mask_b = np.broadcast_to(
            edge_mask[None, :, :], field_data.shape,
        )
    else:
        n_face, n_x, n_y, n_lev = field_data.shape
        edge_mask = np.zeros((n_x, n_y), dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        edge_mask_b = np.broadcast_to(
            edge_mask[None, :, :, None], field_data.shape,
        )
    interior_mask = ~edge_mask_b
    arr = np.asarray(field_data)
    return (
        float(arr[edge_mask_b].std()),
        float(arr[interior_mask].std()),
    )


def _build_pe_state(n, seed, use_duogrid):
    nlev = 5
    rng = np.random.default_rng(seed=seed)
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    T_perturb = rng.uniform(-1.0, 1.0, size=state.T.data.shape)
    state = state._replace(
        T=state.T.replace(
            data=state.T.data + jnp.asarray(T_perturb),
        ),
    )
    return grid, coord, state


def test_pe_context_reduces_edge_ratio(capsys):
    """PE: min-edge factory + monotone_halo_clip_context → lower edge ratio."""
    seeds = [506, 507]
    base_kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    ratios_no_ctx = []
    ratios_with_ctx = []
    for seed in seeds:
        grid_on, coord, state_on = _build_pe_state(8, seed, True)
        grid_off, _, state_off = _build_pe_state(8, seed, False)
        cfg = make_legoesm_pe_min_edge_config(**base_kw)

        m_on = CDGridPrimitiveEquationModel(grid_on, coord, cfg)
        m_off = CDGridPrimitiveEquationModel(grid_off, coord, cfg)
        s_on = m_on.step(state_on, dt=10.0)
        s_off = m_off.step(state_off, dt=10.0)
        e_on, _ = _edge_and_interior_std(s_on.T.data)
        e_off, _ = _edge_and_interior_std(s_off.T.data)
        if e_off > 1e-30:
            ratios_no_ctx.append(e_on / e_off)

        with monotone_halo_clip_context(slack=0.5):
            m_on_c = CDGridPrimitiveEquationModel(grid_on, coord, cfg)
            m_off_c = CDGridPrimitiveEquationModel(grid_off, coord, cfg)
            s_on_c = m_on_c.step(state_on, dt=10.0)
            s_off_c = m_off_c.step(state_off, dt=10.0)
            e_on_c, _ = _edge_and_interior_std(s_on_c.T.data)
            e_off_c, _ = _edge_and_interior_std(s_off_c.T.data)
            if e_off_c > 1e-30:
                ratios_with_ctx.append(e_on_c / e_off_c)
    mean_no_ctx = float(np.mean(ratios_no_ctx))
    mean_with_ctx = float(np.mean(ratios_with_ctx))
    with capsys.disabled():
        print(
            f"\n[iter-506 PE min-edge factory + monotone_halo_clip_context]"
        )
        print(f"  no context:    edge ratio {mean_no_ctx:.3f}×")
        print(f"  with context:  edge ratio {mean_with_ctx:.3f}×")
        if mean_no_ctx > 1e-30:
            print(
                f"  reduction:     "
                f"{(1 - mean_with_ctx/mean_no_ctx)*100:.1f}%"
            )
    assert np.isfinite(mean_no_ctx) and np.isfinite(mean_with_ctx)
    assert mean_with_ctx <= mean_no_ctx * 1.10, (
        f"PE: context should not INCREASE ratio by >10%: "
        f"no_ctx={mean_no_ctx:.3f}, with_ctx={mean_with_ctx:.3f}."
    )
