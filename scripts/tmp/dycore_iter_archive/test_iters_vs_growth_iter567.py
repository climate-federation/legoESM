"""FV3_3D iter 567: does heat_source_del2_iters affect 30-step growth?

iter-562: at 10 steps, iters=8 reduces edge_std by 89%.
iter-566: at 30 steps with iters=8, growth is ~5.9× (super-
linear).  Does iters=2 (FV3 default) have similar or different
growth?  If iters=8 only buys early-time reduction but same
late-time growth, the long-term gain is smaller than expected.

Tests
-----

1. ``test_iters_2_vs_8_30step_growth``.
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
from legoesm.grids.halo import make_clipped_scan_step
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


def _build_sbr_state(n=24, U_0=20.0):
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


def test_iters_2_vs_8_30step_growth(capsys):
    results = []
    for iters in [2, 8]:
        results_n = []
        for n_steps in [10, 30]:
            grid, hc, tm, state = _build_sbr_state(n=24)
            kw = dict(
                n_acoustic_substeps=4,
                damp_v=0.030, damp_v_d_con=1.0,
                corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
                div_damp_coeff=1e6, div_damp_d_con=1.0,
                damp_w=0.030, damp_w_d_con=1.0,
                heat_source_del2_iters=iters,
                heat_source_del2_coeff=0.20,
            )
            cfg = make_legoesm_nh_min_edge_config(**kw)
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            scan_step = make_clipped_scan_step(
                m, state, dt=10.0, n_steps=n_steps, slack=0.5,
            )
            final = scan_step(state)
            e = _edge_std(final.theta_prime.data)
            results_n.append((n_steps, e))
        results.append((iters, results_n))
    with capsys.disabled():
        print(
            f"\n[iter-567 iters=2 vs iters=8 over 10 vs 30 steps @ C24 SBR]"
        )
        for iters, rn in results:
            print(f"\n  iters={iters}:")
            e10 = rn[0][1]
            e30 = rn[1][1]
            growth_factor = e30 / e10 if e10 > 0 else float("nan")
            print(f"    10 steps: edge_std = {e10:.3e}")
            print(f"    30 steps: edge_std = {e30:.3e}")
            print(f"    growth factor: {growth_factor:.2f}×")
    assert all(np.isfinite(e) for _, rn in results for _, e in rn)
