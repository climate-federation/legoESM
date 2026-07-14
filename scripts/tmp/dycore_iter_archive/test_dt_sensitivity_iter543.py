"""FV3_3D iter 543: clip helper robustness across dt values.

iter-526 ``make_clipped_step`` is parameterized on dt at
trace time.  Verify it works correctly at multiple dt values
(traces a different graph per dt) and remains stable.

Tests
-----

1. ``test_dt_sensitivity`` — same model, multiple dt values
   (5, 10, 20 s).  All produce finite states; final state
   should be roughly similar for small dt → no wild divergence.
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
    u_p = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev))
    v_p = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev))
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


def test_dt_sensitivity(capsys):
    """make_clipped_step at dt ∈ {5, 10, 20} — all finite."""
    dts = [5.0, 10.0, 20.0]
    results = []
    for dt in dts:
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
        step = make_clipped_step(m, state, dt=dt, slack=0.5)
        s = state
        # n_steps so that t_final = 100 s in every case
        n_steps = int(round(100.0 / dt))
        for _ in range(n_steps):
            s = step(s, dt)
        # max field magnitudes
        results.append((
            dt, n_steps,
            float(jnp.abs(s.u.data).max()),
            float(jnp.abs(s.theta_prime.data).max()),
        ))
    with capsys.disabled():
        print(
            f"\n[iter-543 dt sensitivity, t_final = 100 s, C8 SBR]"
        )
        print(
            f"  {'dt':>5s}  {'n_steps':>8s}  {'max|u|':>10s}  "
            f"{'max|θ′|':>10s}"
        )
        for dt, n, u_m, t_m in results:
            print(f"  {dt:5.1f}  {n:8d}  {u_m:10.3e}  {t_m:10.3e}")
    # All should be finite
    for dt, n_steps, u_m, t_m in results:
        assert np.isfinite(u_m), f"u became non-finite at dt={dt}"
        assert np.isfinite(t_m), f"theta_prime non-finite at dt={dt}"
