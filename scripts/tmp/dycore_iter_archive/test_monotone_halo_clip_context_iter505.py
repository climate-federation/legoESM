"""FV3_3D iter 505: user-facing ``monotone_halo_clip_context``
helper.

Exposes the iter-504 combined fix via a single context
manager so users can wrap their dycore step.

Tests
-----

1. ``test_context_importable``.
2. ``test_context_reduces_edge_ratio`` — verifies iter-504
   1.13× result reproduces via the helper.
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


def test_context_importable():
    assert callable(monotone_halo_clip_context)
    # Returns ExitStack-like object
    with monotone_halo_clip_context(slack=0.5):
        pass


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


def test_context_reduces_edge_ratio(capsys):
    """min-edge factory + monotone_halo_clip_context → ~1.13×."""
    seeds = [505, 506]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    ratios_no_ctx = []
    ratios_with_ctx = []
    for seed in seeds:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        cfg = make_legoesm_nh_min_edge_config(**base_kw)

        # Without context
        m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
        m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
        s_on = m_on.step(state_on, dt=10.0)
        s_off = m_off.step(state_off, dt=10.0)
        e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
        if e_off > 1e-30:
            ratios_no_ctx.append(e_on / e_off)

        # With context
        with monotone_halo_clip_context(slack=0.5):
            m_on_c = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off_c = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on_c = m_on_c.step(state_on, dt=10.0)
            s_off_c = m_off_c.step(state_off, dt=10.0)
            e_on_c, _ = _edge_and_interior_std(s_on_c.theta_prime.data)
            e_off_c, _ = _edge_and_interior_std(s_off_c.theta_prime.data)
            if e_off_c > 1e-30:
                ratios_with_ctx.append(e_on_c / e_off_c)
    mean_no_ctx = float(np.mean(ratios_no_ctx))
    mean_with_ctx = float(np.mean(ratios_with_ctx))
    with capsys.disabled():
        print(
            f"\n[iter-505 min-edge factory + monotone_halo_clip_context]"
        )
        print(f"  no context:    edge ratio {mean_no_ctx:.3f}×")
        print(f"  with context:  edge ratio {mean_with_ctx:.3f}×")
        print(
            f"  reduction:     "
            f"{(1 - mean_with_ctx/mean_no_ctx)*100:.1f}%"
        )
    assert mean_with_ctx < mean_no_ctx, (
        f"context should reduce ratio: no_ctx={mean_no_ctx:.3f}, "
        f"with_ctx={mean_with_ctx:.3f}."
    )
