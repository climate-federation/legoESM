"""FV3_3D iter 481: sweep ``corner_div_damp_d2_bg`` to find
the value that minimizes the duogrid edge penalty.

iter-480 showed corner_div_damp is THE key mitigator: with it
off the penalty is 109×; with it at d2_bg=5e-4 the penalty
is 3.5×.  Question: does tuning d2_bg HIGHER further reduce
the penalty toward 1×?

Sweep d2_bg in {1e-4, 5e-4 (current factory), 1e-3, 5e-3,
1e-2, 5e-2} at C8 + duogrid.

Tests
-----

1. ``test_corner_div_d2_bg_sweep_duogrid``.
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


def test_corner_div_d2_bg_sweep_duogrid(capsys):
    seeds = [481, 482]   # 2 seeds × 6 values × 2 grids = 24 builds
    sweep_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    common = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    results = {}
    for d2_bg in sweep_values:
        ratios = []
        for seed in seeds:
            grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
            grid_off, _, _, state_off = _build_nh_state(8, seed, False)
            kw = dict(**common, corner_div_damp_d2_bg=d2_bg)
            cfg = make_fv3_faithful_nh_config(**kw)
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on = m_on.step(state_on, dt=10.0)
            s_off = m_off.step(state_off, dt=10.0)
            e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
            e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
            if e_off > 1e-30 and np.isfinite(e_on) and np.isfinite(e_off):
                ratios.append(e_on / e_off)
            else:
                ratios.append(np.nan)
        results[d2_bg] = float(np.nanmean(ratios)) if any(
            not np.isnan(r) for r in ratios
        ) else float("nan")
    with capsys.disabled():
        print(
            f"\n[iter-481 NH corner_div_damp_d2_bg sweep at duogrid, "
            f"2 seeds @ C8, 1 step]"
        )
        for d2_bg, r in results.items():
            print(
                f"  d2_bg = {d2_bg:.0e}: edge ratio = {r:.3f}×"
            )
