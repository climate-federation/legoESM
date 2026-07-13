"""FV3_3D iter 573: NH SBR at low U_0 — how does edge_std scale?

iter-521-569 used U_0=20 m/s.  At lower amplitude (e.g.,
U_0=2 m/s), does edge_std scale linearly with U_0 (suggesting
artifact ∝ flow), or non-linearly?

Tests
-----

1. ``test_low_amplitude_sbr_scaling`` — U_0 ∈ {2, 5, 10, 20}.
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


def _edge_std(field_data):
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    return float(np.asarray(field_data)[edge_mask_b].std())


def _build_sbr_state(n=16, U_0=20.0):
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


def test_low_amplitude_sbr_scaling(capsys):
    results = []
    for U_0 in [2.0, 5.0, 10.0, 20.0]:
        grid, hc, tm, state = _build_sbr_state(n=16, U_0=U_0)
        kw = dict(
            n_acoustic_substeps=4,
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
            heat_source_del2_iters=8,
            heat_source_del2_coeff=0.20,
        )
        cfg = make_legoesm_nh_min_edge_config(**kw)
        m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        step = make_clipped_step(m, state, dt=10.0, slack=0.5)
        s = state
        for _ in range(10):
            s = step(s, 10.0)
        e = _edge_std(s.theta_prime.data)
        results.append((U_0, e))
    with capsys.disabled():
        print(
            f"\n[iter-573 NH SBR low-U_0 scan @ C16, 10 steps, iters=8]"
        )
        print(f"  {'U_0':>5s}: {'edge_std':>10s}  edge/U_0²")
        for U_0, e in results:
            print(f"  {U_0:5.1f}: {e:10.3e}  {e/U_0**2:.3e}")
        # Check if edge_std ~ U_0^p
        if all(e > 0 for _, e in results):
            us = np.array([r[0] for r in results])
            es = np.array([r[1] for r in results])
            p = np.polyfit(np.log(us), np.log(es), 1)
            print(f"\n  Fit: edge_std ~ U_0^{p[0]:.2f}")
    assert all(np.isfinite(e) for _, e in results)
