"""FV3_3D iter 521: solid-body rotation IC (Williamson 2-like).

iter-519 showed cube-smooth Gaussian theta_prime IC converges
(C8 1.43× → C16 1.29× e/i).  Solid-body rotation (SBR) is the
canonical "should stay steady forever" benchmark: u_east =
U₀ cos(lat), v_north = 0.  Any drift from steady state at
cube edges is an edge artifact.

Tests
-----

1. ``test_sbr_resolution_scan`` — SBR initial winds, zero
   theta_prime perturbation.  Measure theta_prime edge/interior
   ratio at 10 steps at C8 and C16.  Since theta_prime starts
   at zero, any nonzero edge bias is purely numerical.
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
    # SBR geographic winds
    u_east = U_0 * jnp.cos(lat)        # (6, n, n)
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


def test_sbr_resolution_scan(capsys):
    results = []
    for n in [8, 16]:
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
        # theta_prime starts at 0; e, i now measure noise generated
        # by the dycore at edges vs interior.
        ratio = e / i if i > 1e-30 else float("nan")
        # Also report absolute magnitudes
        results.append((n, ratio, e, i))
    with capsys.disabled():
        print(
            f"\n[iter-521 SBR IC (Williamson 2-like) @ 10 steps]"
        )
        print(
            f"  {'N':>3s}: {'e/i':>8s}  {'edge_std':>10s}  "
            f"{'interior_std':>13s}"
        )
        for n, r, e, i in results:
            print(
                f"  C{n:2d}: {r:7.3f}×  {e:10.3e}  {i:12.3e}"
            )
        if all(np.isfinite(r) for _, r, _, _ in results):
            r8 = results[0][1]
            r16 = results[1][1]
            print(
                f"\n  C8/C16: {r8/r16:.2f}× ({(1 - r16/r8)*100:+.0f}% at C16)"
            )
            if r16 < r8 * 0.9:
                print(f"  → SBR converges")
            else:
                print(f"  → SBR plateaus")
    assert all(np.isfinite(r) for _, r, _, _ in results)
