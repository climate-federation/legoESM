"""FV3_3D iter 469: PE-specific per-flag edge-ratio sweep
(mirror of NH iter-465).

iter-465 measured NH per-flag impact at C8 + duogrid.  PE may
have DIFFERENT hurting flags due to different staggering /
prognostic variables.  iter-468 added a PE min-edge factory
that mirrored NH iter-467; iter-469 validates whether the
same flag set actually helps PE.

Methodology: toggle each FV3-fidelity flag OFF individually
from the PE factory + duogrid + measure T edge ratio.

Tests
-----

1. ``test_pe_per_flag_edge_ratio_sweep``.
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _edge_interior_ratio(field_data: jnp.ndarray) -> float:
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
    if arr[interior_mask].std() == 0.0:
        return 0.0
    return float(arr[edge_mask_b].std() / arr[interior_mask].std())


def _build_pe_state(n, seed):
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state = hydrostatic_to_fv3(held_suarez_init(grid, coord), cdgrid)
    rng = np.random.default_rng(seed=seed)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def test_pe_per_flag_edge_ratio_sweep(capsys):
    seeds = [469, 470, 471]
    base_kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    overrides_to_test = {
        "baseline (factory)": {},
        "no a2b_zeta_corner": dict(use_fv3_a2b_zeta_corner=False),
        "no metric_aware_d_con": dict(use_fv3_metric_aware_d_con=False),
        "no cross_face_du_proj": dict(use_fv3_cross_face_du_proj=False),
        "no heat_source_del2": dict(heat_source_del2_iters=0),
        "no d_con_top_zero": dict(d_con_top_zero_levels=0),
        "no sponge_damp_v": dict(use_fv3_sponge_damp_v=False),
    }
    results = {}
    for name, overrides in overrides_to_test.items():
        ratios = []
        for seed in seeds:
            grid, coord, state = _build_pe_state(8, seed=seed)
            cfg_kw = {**base_kw, **overrides}
            cfg = make_fv3_faithful_pe_config(**cfg_kw)
            m = CDGridPrimitiveEquationModel(grid, coord, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            ratios.append(_edge_interior_ratio(s.T.data))
        results[name] = float(np.mean(ratios))

    baseline = results["baseline (factory)"]
    with capsys.disabled():
        print(
            f"\n[iter-469 PE per-flag T edge-ratio sweep, "
            f"3 seeds @ C8 + duogrid, 3 steps]"
        )
        print(f"  baseline (factory): {baseline:.4f}")
        print(f"  Per-flag deltas (flag OFF − baseline):")
        for name, r in results.items():
            if name == "baseline (factory)":
                continue
            delta = r - baseline
            sign = "+" if delta >= 0 else ""
            print(f"    {name:30s}: {r:.4f} ({sign}{delta:.4f})")
        print(
            "  POSITIVE delta = flag was REDUCING the ratio "
            "(useful for edge-artifact reduction)."
        )
    assert all(np.isfinite(r) for r in results.values())
