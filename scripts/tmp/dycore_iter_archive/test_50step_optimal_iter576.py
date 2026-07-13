"""FV3_3D iter 576: 50-step SBR with optimal stack.

iter-566 found 30-step C24 → edge_std 34 mK (5.9× growth
from 10 steps).  Extend to 50 steps — does it stay bounded?

Tests
-----

1. ``test_50step_optimal_bounded``.
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


def test_50step_optimal_bounded(capsys):
    n = 24
    n_steps = 50
    grid, hc, tm, state = _build_sbr_state(n=n)
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
    scan_step = make_clipped_scan_step(
        m, state, dt=10.0, n_steps=n_steps, slack=0.5,
    )
    final = scan_step(state)
    e = _edge_std(final.theta_prime.data)
    max_t = float(jnp.abs(final.theta_prime.data).max())
    max_u = float(jnp.abs(final.u.data).max())
    with capsys.disabled():
        print(
            f"\n[iter-576 50-step optimal @ C{n} SBR + iters=8]"
        )
        print(f"  edge_std:    {e:.3e}")
        print(f"  max|θ′|:     {max_t:.3e}")
        print(f"  max|u|:      {max_u:.3e}")
        print(f"\n  Compare:")
        print(f"  iter-564 @ 10 steps: edge_std=5.83e-3, max|u|≈30")
        print(f"  iter-566 @ 30 steps: edge_std=3.43e-2")
        if e > 0:
            growth = e / 5.83e-3
            print(f"\n  growth factor 10→50 steps: {growth:.2f}×")
    assert jnp.all(jnp.isfinite(final.theta_prime.data))
    assert max_u < 100.0, f"u went unphysical: {max_u}"
