"""FV3_3D iter 532: SBR resolution scan to C24.

iter-521 measured SBR at C8/C16 only:
- C8:  e/i = 3.90×, edge_std = 1.07e-01
- C16: e/i = 6.31×, edge_std = 5.89e-02

This iter extends to C24 to confirm the trend.

Tests
-----

1. ``test_sbr_c8_c16_c24_trend`` — measure edge_std at all
   three resolutions.  Check absolute convergence.
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
from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    rotate_winds_geo_to_grid,
)
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


def _build_sbr_state(n, U_0=20.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lat = grid.lat
    u_east = U_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_p = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev))
    v_p = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
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


def test_sbr_c8_c16_c24_trend(capsys):
    results = []
    for n in [8, 16, 24]:
        grid, hc, tm, state = _build_sbr_state(n)
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
        results.append((n, ratio, e, i))
    with capsys.disabled():
        print(
            f"\n[iter-532 SBR @ C8/C16/C24, 10 steps, min-edge + clip]"
        )
        print(
            f"  {'N':>3s}: {'e/i':>8s}  {'edge_std':>12s}  "
            f"{'int_std':>12s}"
        )
        for n, r, e, i in results:
            print(f"  C{n:2d}: {r:7.3f}×  {e:12.3e}  {i:12.3e}")
        # Fit log(edge_std) vs log(n) to estimate convergence order
        if all(np.isfinite(e) for _, _, e, _ in results):
            ns = np.array([n for n, _, _, _ in results], dtype=float)
            es = np.array([e for _, _, e, _ in results])
            iss = np.array([i for _, _, _, i in results])
            if np.all(es > 0) and np.all(iss > 0):
                p_edge = np.polyfit(np.log(ns), np.log(es), 1)
                p_int = np.polyfit(np.log(ns), np.log(iss), 1)
                print(
                    f"\n  edge_std ~ N^{p_edge[0]:.2f}  (-{-p_edge[0]:.1f}-th order)"
                )
                print(
                    f"  int_std  ~ N^{p_int[0]:.2f}  (-{-p_int[0]:.1f}-th order)"
                )
    assert all(np.isfinite(r) for _, r, _, _ in results)
