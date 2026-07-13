"""FV3_3D iter 512: long-term slack sensitivity for
``monotone_halo_clip_context``.

iter-503 at 1 step: slack=0.0 → 2.079×, slack=0.5 → 2.006×
(slack=0.5 slightly better).  iter-511 at 10 steps: slack=0.5
→ 1.793×.  Question: does strict slack=0.0 keep a tighter
long-term bound, or does the slight 1-step penalty grow?

Tests
-----

1. ``test_long_term_slack_sweep`` — measure 10-step ratio
   for slack ∈ {0.0, 0.5, 1.0, 2.0}.
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


def test_long_term_slack_sweep(capsys):
    seed = 512
    n_steps = 10
    slacks = [0.0, 0.5, 1.0, 2.0]
    results = []
    for slack in slacks:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        kw = dict(
            n_acoustic_substeps=4,
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
        )
        cfg = make_legoesm_nh_min_edge_config(**kw)
        with monotone_halo_clip_context(slack=slack):
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on = state_on
            s_off = state_off
            for _ in range(n_steps):
                s_on = m_on.step(s_on, dt=10.0)
                s_off = m_off.step(s_off, dt=10.0)
        e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
        if e_off > 1e-30 and np.isfinite(e_on):
            results.append((slack, e_on / e_off))
        else:
            results.append((slack, float("nan")))
    with capsys.disabled():
        print(
            f"\n[iter-512 long-term slack sweep "
            f"(seed=512, C8, 10 steps)]"
        )
        print(f"  {'slack':>6s}: {'ratio':>8s}")
        for slack, r in results:
            print(f"  {slack:6.2f}: {r:7.3f}×")
        finite = [(s, r) for s, r in results if np.isfinite(r)]
        if finite:
            best = min(finite, key=lambda x: x[1])
            print(f"\n  Best:  slack={best[0]:.2f} → {best[1]:.3f}×")
    assert any(np.isfinite(r) for _, r in results)
