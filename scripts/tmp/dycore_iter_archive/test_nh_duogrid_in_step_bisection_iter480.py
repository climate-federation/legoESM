"""FV3_3D iter 480: bisect in-step duogrid ops.

iter-479 ruled out post-step damp_v / damp_w.  Now isolate
the in-step damping mechanisms by disabling each individually
while keeping others on for stability.

In-step ops (in slow_tendencies):
  * corner_div_damp_d2_bg (B-grid corner-divergence damp)
  * div_damp_coeff (cell-centre divergence damping)
  * hyperdiff_coeff (cell-centre del-4 hyperdiffusion)

Config sweep:
  A. all on (factory baseline)
  B. -corner_div_damp
  C. -div_damp
  D. -hyperdiff
  E. -all three (would diverge; SKIP per iter-479)

Tests
-----

1. ``test_nh_duogrid_in_step_op_bisection``.
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
from legoesm.grids.cubed_sphere import create_cubed_sphere
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


def _build_nh_state(n, seed, use_duogrid):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
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


def test_nh_duogrid_in_step_op_bisection(capsys):
    seeds = [480, 481, 482]
    common = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    full_kw = dict(**common,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    configs = {
        "A. factory baseline": full_kw,
        "B. -corner_div": dict(**common,
            corner_div_damp_d2_bg=0.0, corner_div_damp_d_con=0.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
        ),
        "C. -div_damp": dict(**common,
            corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
            div_damp_coeff=0.0, div_damp_d_con=0.0,
        ),
    }
    results = {}
    for name, kw in configs.items():
        ratios = []
        for seed in seeds:
            grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
            grid_off, _, _, state_off = _build_nh_state(8, seed, False)
            cfg = make_fv3_faithful_nh_config(**kw)
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on = m_on.step(state_on, dt=10.0)
            s_off = m_off.step(state_off, dt=10.0)
            e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
            e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
            if e_off > 1e-30:
                ratios.append(e_on / e_off)
            else:
                ratios.append(1.0)
        results[name] = float(np.mean(ratios))
    with capsys.disabled():
        print(
            f"\n[iter-480 NH duogrid in-step op bisection, "
            f"1 step @ C8]"
        )
        for name, r in results.items():
            print(f"  {name:30s}: edge ratio = {r:.3f}×")
    assert all(np.isfinite(r) for r in results.values())
