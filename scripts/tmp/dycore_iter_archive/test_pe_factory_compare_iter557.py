"""FV3_3D iter 557: PE FV3-faithful vs min-edge comparison.

iter-553 NH finding: at C16 SBR, FV3-faithful BEATS min-edge
(-89.5% vs -64% edge_std reduction).  Does the same hold for
PE?  iter-469 at C8 random IC found PE insensitive to flags
— but smooth IC may differ.

Tests
-----

1. ``test_pe_factory_compare_smooth_ic`` — Held-Suarez init
   + 10 steps, compare PE faithful vs min-edge T edge_std.
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
    make_legoesm_pe_min_edge_config,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _edge_std(field_data):
    if field_data.ndim == 3:
        # (face, x, y)
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


def test_pe_factory_compare_smooth_ic(capsys):
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    results = []
    for label, factory in [
        ("FV3-faithful", make_fv3_faithful_pe_config),
        ("min-edge",     make_legoesm_pe_min_edge_config),
    ]:
        cfg = factory(**kw)
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        step = jax.jit(m.step)
        s = state
        for _ in range(10):
            s = step(s, 10.0)
        e_T = _edge_std(s.T.data)
        results.append((label, e_T))
    with capsys.disabled():
        print(
            f"\n[iter-557 PE factory compare @ C16 Held-Suarez, 10 steps]"
        )
        for label, e in results:
            print(f"  {label:14s}: T edge_std = {e:.3e}")
        if results[0][1] > 0:
            ratio = results[1][1] / results[0][1]
            diff = (results[1][1] - results[0][1]) / results[0][1] * 100
            print(
                f"\n  min-edge / FV3-faithful: {ratio:.3f}× ({diff:+.1f}%)"
            )
    assert all(np.isfinite(e) for _, e in results)
