"""FV3_3D iter 555: does clip helper add value to FV3-faithful factory?

iter-553 found ``make_fv3_faithful_nh_config()`` alone gives
89.5% edge_std reduction at C16 SBR — better than iter-466
min-edge.  Question: does adding the iter-505 clip helper
on top of FV3-faithful improve further?

Tests
-----

1. ``test_faithful_with_vs_without_clip``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
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


def test_faithful_with_vs_without_clip(capsys):
    grid, hc, tm, state = _build_sbr_state(n=16)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_fv3_faithful_nh_config(**kw)

    # Without clip
    m_no_clip = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_no_clip = jax.jit(m_no_clip.step)
    s_no_clip = state
    for _ in range(10):
        s_no_clip = step_no_clip(s_no_clip, 10.0)
    e_no, i_no = _edge_and_interior_std(s_no_clip.theta_prime.data)

    # With clip
    m_clip = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_clip = make_clipped_step(m_clip, state, dt=10.0, slack=0.5)
    s_clip = state
    for _ in range(10):
        s_clip = step_clip(s_clip, 10.0)
    e_c, i_c = _edge_and_interior_std(s_clip.theta_prime.data)

    with capsys.disabled():
        print(
            f"\n[iter-555 FV3-faithful with vs without clip @ C16 SBR]"
        )
        print(
            f"  no clip: edge_std = {e_no:.3e}, int_std = {i_no:.3e}"
        )
        print(
            f"  clip:    edge_std = {e_c:.3e}, int_std = {i_c:.3e}"
        )
        if e_no > 0:
            red = (1 - e_c / e_no) * 100
            print(f"  clip reduction: {red:+.1f}%")
    assert np.isfinite(e_c) and np.isfinite(e_no)
