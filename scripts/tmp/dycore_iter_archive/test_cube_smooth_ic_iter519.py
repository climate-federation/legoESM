"""FV3_3D iter 519: truly cube-smooth IC via geographic coords.

iter-517's "smooth" IC was smooth-on-face but discontinuous
at panel boundaries (iter-518 revealed C16 worse than C8).

A truly cube-smooth IC must be defined in geographic (lat,
lon) coordinates and projected to face-local (u, v) — that's
inherently continuous across panels.

This iter uses a Gaussian theta_prime bump centered at
equator/(λ=π/4), with u=v=0 (rest atmosphere).  No panel-
boundary discontinuity.

Tests
-----

1. ``test_cube_smooth_ic_resolution_scan`` — measure e/i at
   C8 + C16.  If e/i converges (drops with N), the dycore
   is well-behaved on real ICs.
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


def _build_cube_smooth_state(n):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lon = np.asarray(grid.lon)  # (6, n, n) radians
    lat = np.asarray(grid.lat)  # (6, n, n)
    # Gaussian bump centered at λ₀=π/4, φ₀=0 on the sphere
    lon0 = np.pi / 4.0
    lat0 = 0.0
    sigma = 0.5  # ~28° width
    dlon = np.fmod(lon - lon0 + 3 * np.pi, 2 * np.pi) - np.pi
    dist2 = dlon ** 2 * np.cos(lat) ** 2 + (lat - lat0) ** 2
    theta_bump_2d = 5.0 * np.exp(-dist2 / (sigma ** 2))  # (6, n, n)
    theta_prime_p = np.broadcast_to(
        theta_bump_2d[..., None], (6, n, n, nlev),
    ).copy()
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.asarray(theta_prime_p),
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


def test_cube_smooth_ic_resolution_scan(capsys):
    results = []
    for n in [8, 16]:
        grid, hc, tm, state = _build_cube_smooth_state(n)
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
        results.append((n, ratio))
    with capsys.disabled():
        print(
            f"\n[iter-519 cube-smooth IC (geographic Gaussian) @ 10 steps]"
        )
        for n, r in results:
            print(f"  C{n:2d}:  e/i = {r:.3f}×")
        if all(np.isfinite(r) for _, r in results):
            r8 = results[0][1]
            r16 = results[1][1]
            print(
                f"\n  C8/C16: {r8/r16:.2f}× ({(1-r16/r8)*100:+.0f}% at C16)"
            )
            if r16 < r8 * 0.7:
                print(f"  → strong convergence")
            elif r16 < r8 * 0.95:
                print(f"  → modest convergence")
            else:
                print(f"  → no convergence / plateau")
    assert all(np.isfinite(r) for _, r in results)
