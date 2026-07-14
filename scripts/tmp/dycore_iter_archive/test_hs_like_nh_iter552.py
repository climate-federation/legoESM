"""FV3_3D iter 552: Held-Suarez-like stratified NH atmosphere.

iters 506-551 used either flat (zero) or random IC.  This iter
tests with a realistic stratified atmosphere: SBR winds +
Held-Suarez-style meridional temperature gradient (θ' = a
function of latitude).

Tests
-----

1. ``test_hs_like_state_runs_stable`` — 10 steps at C8,
   verify fields stay finite and physical.
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


def _build_hs_like_state(n=8, U_0=20.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lat = grid.lat
    # SBR winds
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
    # HS-like meridional T gradient: θ' = -delta_T * sin²(lat)
    # mimicking pole-equator gradient.  Each level same pattern.
    delta_T = 10.0  # K equator-pole
    theta_2d = -delta_T * jnp.sin(lat) ** 2
    theta_p = jnp.broadcast_to(
        theta_2d[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=theta_p,
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


def test_hs_like_state_runs_stable(capsys):
    grid, hc, tm, state = _build_hs_like_state(n=8)
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
    s = state
    for _ in range(10):
        s = step(s, 10.0)
    with capsys.disabled():
        print(
            f"\n[iter-552 HS-like NH (SBR + θ' gradient), 10 steps @ C8]"
        )
        for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
            a = np.asarray(getattr(s, fld).data)
            print(
                f"  {fld:12s}: min={a.min():.3e}, max={a.max():.3e}, "
                f"std={a.std():.3e}"
            )
    for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
        d = getattr(s, fld).data
        assert jnp.all(jnp.isfinite(d)), (
            f"{fld} non-finite at 10 steps with HS-like IC"
        )
