"""FV3_3D iter 479: bisect single-step duogrid edge-std penalty
by progressively DISABLING dycore subops.

iter-478 showed the 3.67×+ edge-std penalty fires in step 1.
Single-step ops:
  A. slow_tendencies (compute du/dt, dθ/dt, etc.)
  B. RK3 outer integration
  C. Acoustic substeps (default 4 inside RK3)
  D. Post-step corner_div damping
  E. Post-step damp_v
  F. Post-step damp_w
  G. fix_mass

This test progressively disables D,E,F to find which post-
step op contributes most to the duogrid edge penalty.

Configurations:
  A. factory (all ops on)
  B. A + damp_v=0 (no vorticity damping)
  C. A + damp_w=0 + damp_v=0
  D. A + corner_div_damp_d2_bg=0 + damp_v=0 + damp_w=0

Tests
-----

1. ``test_nh_duogrid_op_bisection``.
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


def _run_one_step(grid, hc, tm, state, cfg_kw):
    cfg = make_fv3_faithful_nh_config(**cfg_kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = m.step(state, dt=10.0)
    return s


def test_nh_duogrid_op_bisection(capsys):
    seeds = [479, 480, 481]
    configs = {
        "A. factory baseline": dict(
            n_acoustic_substeps=4,
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
        ),
        "B. -damp_v": dict(
            n_acoustic_substeps=4,
            damp_v=0.0, damp_v_d_con=0.0,
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
        ),
        "C. -damp_v -damp_w": dict(
            n_acoustic_substeps=4,
            damp_v=0.0, damp_v_d_con=0.0,
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.0, damp_w_d_con=0.0,
        ),
        "D. -all damping": dict(
            n_acoustic_substeps=4,
            damp_v=0.0, damp_v_d_con=0.0,
            corner_div_damp_d2_bg=0.0,
            corner_div_damp_d_con=0.0,
            div_damp_coeff=0.0, div_damp_d_con=0.0,
            damp_w=0.0, damp_w_d_con=0.0,
        ),
    }
    results = {}
    for name, kw in configs.items():
        ratios = []
        for seed in seeds:
            grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
            grid_off, _, _, state_off = _build_nh_state(8, seed, False)
            s_on = _run_one_step(grid_on, hc, tm, state_on, kw)
            s_off = _run_one_step(grid_off, hc, tm, state_off, kw)
            e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
            e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
            if e_off > 1e-30:
                ratios.append(e_on / e_off)
            else:
                ratios.append(1.0)
        results[name] = float(np.mean(ratios))
    with capsys.disabled():
        print(
            f"\n[iter-479 NH duogrid op bisection, 1 step @ C8]"
        )
        for name, r in results.items():
            print(f"  {name:30s}: edge ratio = {r:.3f}×")
    assert all(np.isfinite(r) for r in results.values())
