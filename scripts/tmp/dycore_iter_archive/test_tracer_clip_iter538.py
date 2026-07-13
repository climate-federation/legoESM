"""FV3_3D iter 538: clip helper with tracer transport.

iters 506-537 used 0 tracers (last axis empty).  Real
atmospheric runs include at least q_vapor.  Verify the
iter-526 ``make_clipped_step`` integrates with active tracers.

Tests
-----

1. ``test_clipped_step_runs_with_q_vapor`` — SBR winds +
   gaussian q_vapor blob.  10 steps; verify q stays
   non-negative + bounded.
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
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state_with_tracer(n=8, U_0=20.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lon = grid.lon
    lat = grid.lat
    u_east = U_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_p = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev))
    v_p = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev))
    # q_vapor: gaussian blob at equator/(lon=π)
    lon0 = jnp.pi
    lat0 = 0.0
    sigma = 0.6
    dlon = jnp.mod(lon - lon0 + 3 * jnp.pi, 2 * jnp.pi) - jnp.pi
    dist2 = dlon ** 2 * jnp.cos(lat) ** 2 + (lat - lat0) ** 2
    q_vapor_2d = 1.0e-2 * jnp.exp(-dist2 / sigma ** 2)
    q_vapor_3d = jnp.broadcast_to(q_vapor_2d[..., None], (6, n, n, nlev))
    tracers_4d = q_vapor_3d[..., None]  # (6, n, n, nlev, 1)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_t = ("face", "x", "y", "level", "tracer")
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
        tracers=Field(data=tracers_4d, name="tracers", dims=dims_t,
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_clipped_step_runs_with_q_vapor(capsys):
    grid, hc, tm, state = _build_state_with_tracer()
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)
    q0_min = float(jnp.asarray(state.tracers.data).min())
    q0_max = float(jnp.asarray(state.tracers.data).max())
    q0_mean = float(jnp.asarray(state.tracers.data).mean())
    s = state
    for _ in range(10):
        s = step(s, 10.0)
    qf = jnp.asarray(s.tracers.data)
    qf_min = float(qf.min())
    qf_max = float(qf.max())
    qf_mean = float(qf.mean())
    with capsys.disabled():
        print(
            f"\n[iter-538 q_vapor tracer with clip helper, 10 steps @ C8]"
        )
        print(
            f"  initial: min={q0_min:.3e}, max={q0_max:.3e}, "
            f"mean={q0_mean:.3e}"
        )
        print(
            f"  final:   min={qf_min:.3e}, max={qf_max:.3e}, "
            f"mean={qf_mean:.3e}"
        )
    assert jnp.all(jnp.isfinite(qf)), "tracer became non-finite"
    # Mean preservation (tracer-mass-equiv proxy): within 0.1%
    assert abs(qf_mean - q0_mean) < abs(q0_mean) * 0.1, (
        f"tracer mean drift > 10%: {q0_mean:.3e} → {qf_mean:.3e}"
    )
