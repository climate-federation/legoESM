"""FV3_3D iter 551: helper performance at C16 — does overhead scale?

iter-541 measured +0.6% overhead at C8.  At higher resolution
the absolute step time grows but the clip overhead may scale
differently (more halo cells but same per-call overhead).

Tests
-----

1. ``test_clip_helper_overhead_c16`` — measure at C16.
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
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state(n=16, U_0=20.0):
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
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float64),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0), dtype=jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _time_n_steps(step_fn, state, n_steps):
    s = step_fn(state, 10.0)
    jax.block_until_ready(s.u.data)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        s = step_fn(s, 10.0)
    jax.block_until_ready(s.u.data)
    return (time.perf_counter() - t0) / n_steps


def test_clip_helper_overhead_c16(capsys):
    grid, hc, tm, state = _build_state(n=16)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)

    m_raw = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_raw = jax.jit(m_raw.step)
    _ = step_raw(state, 10.0)
    jax.block_until_ready(_.u.data)

    m_clip = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_clip = make_clipped_step(m_clip, state, dt=10.0, slack=0.5)

    n_steps = 5
    t_raw = _time_n_steps(step_raw, state, n_steps)
    t_clip = _time_n_steps(step_clip, state, n_steps)
    overhead = (t_clip - t_raw) / t_raw * 100
    with capsys.disabled():
        print(
            f"\n[iter-551 clip helper perf @ C16, {n_steps} steps]"
        )
        print(f"  raw jit:           {t_raw*1000:.1f} ms/step")
        print(f"  make_clipped_step: {t_clip*1000:.1f} ms/step")
        print(f"  overhead: {overhead:+.1f}%")
    assert t_clip < t_raw * 5, (
        f"clip overhead too large: t_raw={t_raw*1000:.1f}, "
        f"t_clip={t_clip*1000:.1f}"
    )
