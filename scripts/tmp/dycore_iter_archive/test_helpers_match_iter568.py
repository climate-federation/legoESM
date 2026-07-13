"""FV3_3D iter 568: confirm both clip helpers produce same result.

``monotone_halo_clip_context`` (iter-505) and ``make_clipped_step``
(iter-526) use the same underlying patching mechanism.  They
should produce numerically identical results.

Tests
-----

1. ``test_context_vs_make_clipped_step_match`` — bit-for-bit
   match.
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
from legoesm.grids.halo import (
    make_clipped_step,
    monotone_halo_clip_context,
)
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state(n=8):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lat = grid.lat
    u_east = 20.0 * jnp.cos(lat)
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


def test_context_vs_make_clipped_step_match():
    grid, hc, tm, state = _build_state(n=8)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)

    # Path A: monotone_halo_clip_context + raw step (Python loop)
    m_a = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    with monotone_halo_clip_context(slack=0.5):
        s_a = state
        for _ in range(3):
            s_a = m_a.step(s_a, 10.0)

    # Path B: make_clipped_step (with separate model instance)
    m_b = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_b = make_clipped_step(m_b, state, dt=10.0, slack=0.5)
    s_b = state
    for _ in range(3):
        s_b = step_b(s_b, 10.0)

    # Compare
    for fld in ("u", "v", "theta_prime", "rho_prime", "w"):
        a = getattr(s_a, fld).data
        b = getattr(s_b, fld).data
        diff = float(jnp.abs(a - b).max())
        # Allow small float roundoff difference
        assert diff < 1e-10, (
            f"context vs make_clipped_step diverge on {fld}: "
            f"max|diff|={diff}"
        )
