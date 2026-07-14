"""FV3_3D iter 571: PE time-growth power-law (mirror iter-569).

iter-569 found NH edge_std ~ n_steps^1.15 at C16 SBR.  Does
PE show similar power-law growth?

Tests
-----

1. ``test_pe_edge_std_vs_n_steps_fit``.
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
    make_fv3_faithful_pe_config,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import standard_hybrid_levels


def _edge_std(field_data):
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
    return float(np.asarray(field_data)[edge_mask_b].std())


def test_pe_edge_std_vs_n_steps_fit(capsys):
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    state = state._replace(
        u_d=state.u_d.replace(data=state.u_d.data.astype(jnp.float64)),
        v_d=state.v_d.replace(data=state.v_d.data.astype(jnp.float64)),
        T=state.T.replace(data=state.T.data.astype(jnp.float64)),
        p_s=state.p_s.replace(data=state.p_s.data.astype(jnp.float64)),
        phis=state.phis.replace(data=state.phis.data.astype(jnp.float64)),
    )
    kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        heat_source_del2_iters=8,
        heat_source_del2_coeff=0.20,
    )
    cfg = make_fv3_faithful_pe_config(**kw)
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)
    history = []
    s = state
    initial_T_edge = _edge_std(state.T.data)
    for step_idx in range(1, 31):
        s = step(s, 10.0)
        if step_idx in (1, 5, 10, 20, 30):
            e_T_delta = _edge_std(np.asarray(s.T.data) - np.asarray(state.T.data))
            history.append((step_idx, e_T_delta))
    with capsys.disabled():
        print(
            f"\n[iter-571 PE δT edge_std vs n_steps @ C16 HS + "
            f"iters=8]"
        )
        print(f"  initial T edge_std: {initial_T_edge:.3e}")
        for n_step, e in history:
            print(f"  step {n_step:2d}: δT edge_std = {e:.3e}")
        finite = [(n, e) for n, e in history if e > 0]
        if len(finite) >= 2:
            ns = np.array([n for n, _ in finite], dtype=float)
            es = np.array([e for _, e in finite])
            p = np.polyfit(np.log(ns), np.log(es), 1)
            print(f"\n  PE power law: δT edge_std ~ n_steps^{p[0]:.2f}")
            print(f"  NH (iter-569): edge_std ~ n_steps^1.15")
    assert all(np.isfinite(e) for _, e in history)
