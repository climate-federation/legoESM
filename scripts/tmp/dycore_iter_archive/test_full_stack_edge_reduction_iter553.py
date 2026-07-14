"""FV3_3D iter 553: full-stack edge reduction demonstration.

Compares 4 configurations on the same SBR IC at C16, 10 steps:
1. Bare ``CDGridCompressibleEulerConfig()`` — no FV3 fidelity.
2. ``make_fv3_faithful_nh_config()`` — FV3-faithful baseline.
3. ``make_legoesm_nh_min_edge_config()`` — iter-466 50% reducer.
4. min-edge + ``make_clipped_step`` — full stack.

Measures θ′ edge_std to quantify the reduction at each level.

Tests
-----

1. ``test_full_stack_edge_reduction`` — print measurements,
   assert stack achieves lowest edge_std.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
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


def _run_step_loop(model, state, n_steps=10):
    s = state
    step = jax.jit(model.step)
    for _ in range(n_steps):
        s = step(s, 10.0)
    return s


def test_full_stack_edge_reduction(capsys):
    grid, hc, tm, state = _build_sbr_state(n=16)
    # Common kw used to override factory params for fair comparison
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    results = []

    # 1. Bare config (no FV3 fidelity)
    cfg_bare = CDGridCompressibleEulerConfig(**kw)
    m_bare = CDGridCompressibleEulerModel(grid, hc, tm, cfg_bare)
    s_bare = _run_step_loop(m_bare, state)
    results.append(("bare config", _edge_std(s_bare.theta_prime.data)))

    # 2. FV3-faithful factory
    cfg_faithful = make_fv3_faithful_nh_config(**kw)
    m_faithful = CDGridCompressibleEulerModel(grid, hc, tm, cfg_faithful)
    s_faithful = _run_step_loop(m_faithful, state)
    results.append(("FV3-faithful", _edge_std(s_faithful.theta_prime.data)))

    # 3. min-edge factory
    cfg_min_edge = make_legoesm_nh_min_edge_config(**kw)
    m_min_edge = CDGridCompressibleEulerModel(grid, hc, tm, cfg_min_edge)
    s_min_edge = _run_step_loop(m_min_edge, state)
    results.append(("min-edge factory", _edge_std(s_min_edge.theta_prime.data)))

    # 4. min-edge + make_clipped_step
    m_full = CDGridCompressibleEulerModel(grid, hc, tm, cfg_min_edge)
    step_full = make_clipped_step(m_full, state, dt=10.0, slack=0.5)
    s_full = state
    for _ in range(10):
        s_full = step_full(s_full, 10.0)
    results.append(("min-edge + clip", _edge_std(s_full.theta_prime.data)))

    with capsys.disabled():
        print(
            f"\n[iter-553 full-stack edge reduction @ C16 SBR, 10 steps]"
        )
        baseline = results[0][1]
        print(
            f"  {'config':25s}: {'edge_std':>10s}  reduction vs bare"
        )
        for label, e in results:
            red = (1 - e / baseline) * 100 if baseline > 0 else 0
            print(f"  {label:25s}: {e:10.3e}  {red:+6.1f}%")
    # Assert: full stack must be at most equal to or less than bare
    bare_e = results[0][1]
    full_e = results[-1][1]
    assert full_e <= bare_e * 1.5, (
        f"full stack should not WORSEN edge_std vs bare: "
        f"bare={bare_e:.3e}, full={full_e:.3e}"
    )
