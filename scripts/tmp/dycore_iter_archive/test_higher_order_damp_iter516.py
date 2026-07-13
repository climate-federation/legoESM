"""FV3_3D iter 516: higher-order damp (nord=2) for edge bias.

iter-515 showed:
- duogrid e/i = 2.06× at 10 steps (real artifact)
- no-duo e/i = 1.24× at 10 steps (lesser but still real)

The current min-edge factory uses nord_v=1 +
corner_div_damp_nord=1 (i.e. del² damping).  FV3 production
sometimes uses nord=2 (del⁴) for stronger short-scale damping
which concentrates near corners (high curvature).

Tests
-----

1. ``test_nord_sweep_10_steps`` — sweep
   (nord_v, corner_div_damp_nord) ∈ {1, 2}².  Measure
   duogrid e/i at 10 steps.
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


def _build_nh_state(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
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


def test_nord_sweep_10_steps(capsys):
    seed = 516
    configs = [
        ("nord_v=1, cdd_nord=1 (default)", 1, 1),
        ("nord_v=1, cdd_nord=2", 1, 2),
        ("nord_v=2, cdd_nord=1", 2, 1),
        ("nord_v=2, cdd_nord=2", 2, 2),
    ]
    results = []
    for name, nord_v, cdd_nord in configs:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed)
        kw = dict(
            n_acoustic_substeps=4,
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
            nord_v=nord_v,
            corner_div_damp_nord=cdd_nord,
        )
        cfg = make_legoesm_nh_min_edge_config(**kw)
        with monotone_halo_clip_context(slack=0.5):
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            s_on = state_on
            for _ in range(10):
                s_on = m_on.step(s_on, dt=10.0)
        e_on, i_on = _edge_and_interior_std(s_on.theta_prime.data)
        ratio = e_on / i_on if i_on > 1e-30 else float("nan")
        results.append((name, ratio))
    with capsys.disabled():
        print(
            f"\n[iter-516 (nord_v, corner_div_damp_nord) sweep @ 10 steps]"
        )
        print(f"  {'config':40s}: {'e/i ratio':>10s}")
        for name, r in results:
            print(f"  {name:40s}: {r:9.3f}×")
        finite = [(n, r) for n, r in results if np.isfinite(r)]
        if finite:
            best = min(finite, key=lambda x: x[1])
            print(f"\n  Best:  {best[0]} → {best[1]:.3f}×")
    assert any(np.isfinite(r) for _, r in results)
