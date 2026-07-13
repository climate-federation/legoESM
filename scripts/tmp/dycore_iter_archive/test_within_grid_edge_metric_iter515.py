"""FV3_3D iter 515: reframe the edge-artifact metric.

Iters 466-514 used the **cross-grid** metric:
``ratio = edge_std(duogrid) / edge_std(no_duogrid)``.

This is sensitive to the duogrid vs non-duogrid discretization
difference — two valid grids solving slightly different
equations — not just the "edge artifact" of duogrid.

A more direct metric: **within-grid**
``ratio = edge_std(duogrid) / interior_std(duogrid)``.
If close to 1.0, the duogrid run looks uniform across the
panel.  If >> 1.0, there is a real cube imprint regardless
of what the no-duogrid run does.

Tests
-----

1. ``test_within_grid_edge_metric_at_n_steps`` — measure both
   metrics at 1, 5, 10 steps under min-edge + clip.  Report
   side-by-side.
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


def _build_nh_state(n, seed, use_duogrid):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
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


def test_within_grid_edge_metric_at_n_steps(capsys):
    seed = 515
    n_steps_sweep = [1, 5, 10]
    results = []
    for n_steps in n_steps_sweep:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
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
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on = state_on
            s_off = state_off
            for _ in range(n_steps):
                s_on = m_on.step(s_on, dt=10.0)
                s_off = m_off.step(s_off, dt=10.0)
        e_on, i_on = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, i_off = _edge_and_interior_std(s_off.theta_prime.data)
        cross = e_on / e_off if e_off > 1e-30 else float("nan")
        within_on = e_on / i_on if i_on > 1e-30 else float("nan")
        within_off = e_off / i_off if i_off > 1e-30 else float("nan")
        results.append((n_steps, cross, within_on, within_off))
    with capsys.disabled():
        print(
            f"\n[iter-515 reframe edge metrics (seed=515, C8, "
            f"min-edge + clip)]"
        )
        print(
            f"  {'n_steps':>8s}: {'cross':>8s}  "
            f"{'duogrid e/i':>11s}  {'no-duo e/i':>10s}"
        )
        for n_steps, cross, w_on, w_off in results:
            print(
                f"  {n_steps:8d}: {cross:7.3f}×  "
                f"{w_on:10.3f}×  {w_off:9.3f}×"
            )
        # Within-grid ratio at 10 steps: this is the actual edge bias
        # in the duogrid run.
        w_on_10 = results[-1][2]
        if w_on_10 < 1.5:
            print(
                f"\n  duogrid e/i @ 10 steps = {w_on_10:.3f}× → "
                f"edge looks ~uniform with interior"
            )
        else:
            print(
                f"\n  duogrid e/i @ 10 steps = {w_on_10:.3f}× → "
                f"real edge artifact remains"
            )
    assert all(np.isfinite(c) for _, c, _, _ in results)
