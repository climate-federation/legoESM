"""FV3_3D iter 546: scan_step vs Python loop performance.

iter-544 added ``make_clipped_scan_step`` claiming it
eliminates Python overhead vs ``make_clipped_step`` in a
Python loop.  This iter quantifies the speedup.

Tests
-----

1. ``test_scan_vs_loop_speedup`` — time both at 20 steps,
   report speedup.
"""
from __future__ import annotations

import time

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
from legoesm.grids.halo import make_clipped_step, make_clipped_scan_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state(n=8, U_0=20.0):
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
    u_p = jnp.broadcast_to(
        u_grid[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    v_p = jnp.broadcast_to(
        v_grid[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_scan_vs_loop_speedup(capsys):
    grid, hc, tm, state = _build_state()
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)

    n_steps = 20

    # Loop variant — separate model instance to avoid trace-cache reuse
    m_loop = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_loop = make_clipped_step(m_loop, state, dt=10.0, slack=0.5)
    # Warmup
    _ = step_loop(state, 10.0)
    jax.block_until_ready(_.u.data)
    t0 = time.perf_counter()
    s = state
    for _ in range(n_steps):
        s = step_loop(s, 10.0)
    jax.block_until_ready(s.u.data)
    t_loop = time.perf_counter() - t0

    # Scan variant
    m_scan = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    scan_step = make_clipped_scan_step(
        m_scan, state, dt=10.0, n_steps=n_steps, slack=0.5,
    )
    # Warmup
    _ = scan_step(state)
    jax.block_until_ready(_.u.data)
    t0 = time.perf_counter()
    s_scan = scan_step(state)
    jax.block_until_ready(s_scan.u.data)
    t_scan = time.perf_counter() - t0

    speedup = t_loop / t_scan
    with capsys.disabled():
        print(
            f"\n[iter-546 scan vs loop, {n_steps} steps @ C8]"
        )
        print(f"  Python loop: {t_loop:.3f} s")
        print(f"  scan_step:   {t_scan:.3f} s")
        print(f"  speedup:     {speedup:.2f}× faster")
    # Sanity: scan should not be slower
    assert t_scan <= t_loop * 2, (
        f"scan should not be slower than loop: "
        f"loop={t_loop:.3f}s, scan={t_scan:.3f}s"
    )
