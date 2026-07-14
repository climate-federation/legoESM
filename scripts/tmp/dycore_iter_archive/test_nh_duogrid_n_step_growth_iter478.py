"""FV3_3D iter 478: per-step bisection of iter-471 duogrid
edge-std penalty.

iter-471/472/473 showed duogrid increases NH θ′ edge std
~4.77× after 3 steps.  iter-474..477 exhausted single-
operator bisection.  iter-478 measures the ratio after
1, 2, 3 steps — does the penalty arrive in step 1 or
accumulate?

If single-step ratio ≈ 4.77× → bug is in single-step
dynamics (acoustic substeps + RK3 stages + halo composition).
If single-step ratio ≈ 1× and grows → bug accumulates
through state evolution.

Tests
-----

1. ``test_nh_duogrid_edge_ratio_per_step``.
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


def test_nh_duogrid_edge_ratio_per_step(capsys):
    seeds = [478, 479, 480]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    max_steps = 3
    edge_stds_on = [[] for _ in range(max_steps + 1)]
    edge_stds_off = [[] for _ in range(max_steps + 1)]
    int_stds_on = [[] for _ in range(max_steps + 1)]
    int_stds_off = [[] for _ in range(max_steps + 1)]
    for seed in seeds:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        cfg = make_fv3_faithful_nh_config(**base_kw)
        m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
        m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
        s_on, s_off = state_on, state_off
        e_on, i_on = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, i_off = _edge_and_interior_std(s_off.theta_prime.data)
        edge_stds_on[0].append(e_on)
        edge_stds_off[0].append(e_off)
        int_stds_on[0].append(i_on)
        int_stds_off[0].append(i_off)
        for step in range(1, max_steps + 1):
            s_on = m_on.step(s_on, dt=10.0)
            s_off = m_off.step(s_off, dt=10.0)
            e_on, i_on = _edge_and_interior_std(s_on.theta_prime.data)
            e_off, i_off = _edge_and_interior_std(s_off.theta_prime.data)
            edge_stds_on[step].append(e_on)
            edge_stds_off[step].append(e_off)
            int_stds_on[step].append(i_on)
            int_stds_off[step].append(i_off)

    with capsys.disabled():
        print(
            f"\n[iter-478 NH θ′ edge std growth per step, "
            f"3 seeds @ C8]"
        )
        print(
            f"  step  edge_off    edge_on     ratio   "
            f"interior_off  interior_on"
        )
        for step in range(max_steps + 1):
            me_off = np.mean(edge_stds_off[step])
            me_on = np.mean(edge_stds_on[step])
            mi_off = np.mean(int_stds_off[step])
            mi_on = np.mean(int_stds_on[step])
            ratio = me_on / max(me_off, 1e-30)
            print(
                f"  {step}     {me_off:.4e}  {me_on:.4e}  "
                f"{ratio:.2f}×   {mi_off:.4e}    {mi_on:.4e}"
            )
    assert all(
        np.isfinite(np.mean(edge_stds_off[s]))
        for s in range(max_steps + 1)
    )
    assert all(
        np.isfinite(np.mean(edge_stds_on[s]))
        for s in range(max_steps + 1)
    )
