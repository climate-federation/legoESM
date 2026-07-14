"""FV3_3D iter 544: ``make_clipped_scan_step`` test.

iter-526's ``make_clipped_step`` returns a single-step
JIT-compiled function.  For long runs, calling it in a
Python loop has overhead.  iter-544's ``make_clipped_scan_
step`` compiles the n-step loop as a single ``jax.lax.scan``,
eliminating Python overhead.

Tests
-----

1. ``test_scan_step_matches_loop`` — final state after
   ``scan_step(state)`` matches Python loop of
   ``make_clipped_step`` to high precision.
2. ``test_scan_step_grad`` — ``jax.grad`` flows through.
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


def test_scan_step_matches_loop():
    """scan_step output should match Python-loop step output bit-for-bit."""
    grid, hc, tm, state = _build_state()
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)

    # Python-loop reference
    m_loop = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_loop = make_clipped_step(m_loop, state, dt=10.0, slack=0.5)
    s_loop = state
    for _ in range(5):
        s_loop = step_loop(s_loop, 10.0)

    # scan_step variant
    m_scan = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    scan_step = make_clipped_scan_step(
        m_scan, state, dt=10.0, n_steps=5, slack=0.5,
    )
    s_scan = scan_step(state)

    # Compare fields
    for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
        a = getattr(s_loop, fld).data
        b = getattr(s_scan, fld).data
        diff = float(jnp.abs(a - b).max())
        assert diff < 1e-10, (
            f"scan_step and loop diverge on {fld}: max|diff|={diff}"
        )


def test_scan_step_grad():
    grid, hc, tm, state = _build_state()
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    scan_step = make_clipped_scan_step(
        m, state, dt=10.0, n_steps=3, slack=0.5,
    )

    u0 = state.u.data

    def loss_fn(u_init):
        s = state._replace(u=state.u.replace(data=u_init))
        final = scan_step(s)
        return jnp.mean(final.theta_prime.data ** 2)

    g = jax.grad(loss_fn)(u0)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 1e-12
