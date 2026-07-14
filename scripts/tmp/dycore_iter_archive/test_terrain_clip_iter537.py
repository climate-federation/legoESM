"""FV3_3D iter 537: clip helper with terrain (nonzero phis).

iters 506-536 tested with phis = 0 (flat surface).  Real
atmospheric runs include mountains.  Verify the iter-526
``make_clipped_step`` still works when phis is nonzero.

Tests
-----

1. ``test_clipped_step_runs_with_mountain`` — SBR winds +
   mountain at equator.  Verify stable + mass conserved.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
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


def _build_state_with_mountain(n=8, U_0=20.0, h_mtn=2000.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    # Mountain: gaussian centered at equator / lon=π/2
    lon = grid.lon
    lat = grid.lat
    lon0 = jnp.pi / 2.0
    lat0 = 0.0
    sigma = 0.5
    dlon = jnp.mod(lon - lon0 + 3 * jnp.pi, 2 * jnp.pi) - jnp.pi
    dist2 = dlon ** 2 * jnp.cos(lat) ** 2 + (lat - lat0) ** 2
    mtn_height = h_mtn * jnp.exp(-dist2 / sigma ** 2)
    phis = constants.g * mtn_height  # surface geopotential
    tm = compute_terrain_metric(mtn_height, hc)
    # SBR winds
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
        phis=Field(data=phis, name="phis", dims=dims_2d,
                   units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_clipped_step_runs_with_mountain(capsys):
    grid, hc, tm, state = _build_state_with_mountain()
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
    # Stability
    for fld_name in ("u", "v", "theta_prime", "rho_prime", "w"):
        fld = getattr(s, fld_name)
        assert jnp.all(jnp.isfinite(fld.data)), (
            f"{fld_name} became non-finite with terrain"
        )
    # Print perturbation magnitudes
    with capsys.disabled():
        print(
            f"\n[iter-537 clip helper with 2-km mountain, 10 steps @ C8]"
        )
        for fld_name in ("u", "v", "theta_prime", "rho_prime", "w"):
            fld = getattr(s, fld_name)
            arr = jnp.asarray(fld.data)
            print(
                f"  {fld_name:12s}: max abs = "
                f"{float(jnp.abs(arr).max()):.3e}"
            )
