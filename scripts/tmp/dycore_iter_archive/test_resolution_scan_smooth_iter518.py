"""FV3_3D iter 518: does edge bias converge with resolution?

iter-517 showed smooth IC reduces e/i to 1.26× at C8.
Question: at higher resolution (C16), does the smooth-IC e/i
drop further (true convergence) or plateau (intrinsic
boundary-layer issue at cube edges)?

Convergence → 1st-order solution near edges.
Plateau    → permanent O(1) edge boundary layer.

Tests
-----

1. ``test_resolution_scan_smooth_ic`` — measure smooth-IC e/i
   at C8 and C16, 10 steps each.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _edge_and_interior_std(field_data):
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


def _build_smooth_state(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    i_idx, j_idx = np.meshgrid(
        np.arange(n) / max(n - 1, 1),
        np.arange(n) / max(n - 1, 1),
        indexing="ij",
    )
    base = 3.0 * np.sin(np.pi * i_idx) * np.cos(np.pi * j_idx)
    u_p = np.zeros((6, n, n, nlev))
    v_p = np.zeros((6, n, n, nlev))
    for f in range(6):
        for k in range(nlev):
            u_p[f, :, :, k] = base
            v_p[f, :, :, k] = -base
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_resolution_scan_smooth_ic(capsys):
    seed = 518
    results = []
    for n in [8, 16]:
        grid, hc, tm, state = _build_smooth_state(n, seed)
        kw = dict(
            n_acoustic_substeps=4,
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
        )
        cfg = make_legoesm_nh_min_edge_config(**kw)
        with monotone_halo_clip_context(slack=0.5):
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(10):
                s = m.step(s, dt=10.0)
        e, i = _edge_and_interior_std(s.theta_prime.data)
        ratio = e / i if i > 1e-30 else float("nan")
        results.append((n, ratio))
    with capsys.disabled():
        print(
            f"\n[iter-518 resolution scan smooth IC @ 10 steps "
            f"(seed=518, min-edge + clip)]"
        )
        for n, r in results:
            print(f"  C{n:2d}:  e/i = {r:.3f}×")
        if all(np.isfinite(r) for _, r in results):
            r8 = results[0][1]
            r16 = results[1][1]
            ratio_8_16 = r8 / r16 if r16 > 0 else float("nan")
            print(
                f"\n  C8/C16 ratio: {ratio_8_16:.2f}× "
                f"({(1 - r16/r8)*100:+.0f}% improvement at C16)"
            )
            if r16 < r8 * 0.7:
                print(
                    f"  → strong convergence (>30% improvement)"
                )
            elif r16 < r8 * 0.9:
                print(
                    f"  → modest convergence (10-30% improvement)"
                )
            else:
                print(
                    f"  → plateau (intrinsic edge boundary layer)"
                )
    assert all(np.isfinite(r) for _, r in results)
