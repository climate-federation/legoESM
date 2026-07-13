"""FV3_3D iter 511: does ``monotone_halo_clip_context`` matter
long-term, or just at 1 step?

iter-509 showed the residual grows under min-edge + clip:
1.13 → 1.63 over 10 steps.  Question: does clip vs no-clip
diverge over time, or do they grow in parallel?

If parallel: clip is just a one-step cosmetic — useless long-term.
If diverging: clip's benefit accumulates too (in the right
direction).

Tests
-----

1. ``test_clip_vs_noclip_multi_step`` — measure edge ratio
   at 1, 5, 10 steps for (a) no clip, (b) clip context.
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


def _run_n_steps(seed, n_steps, use_clip):
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
    import contextlib
    ctx = monotone_halo_clip_context(slack=0.5) if use_clip else contextlib.nullcontext()
    with ctx:
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
        return e_on / e_off
    return float("nan")


def test_clip_vs_noclip_multi_step(capsys):
    seed = 511
    n_steps_sweep = [1, 5, 10]
    results = []
    for n_steps in n_steps_sweep:
        r_noclip = _run_n_steps(seed, n_steps, use_clip=False)
        r_clip = _run_n_steps(seed, n_steps, use_clip=True)
        results.append((n_steps, r_noclip, r_clip))
    with capsys.disabled():
        print(
            f"\n[iter-511 clip vs no-clip over n_steps "
            f"(seed=511, C8)]"
        )
        print(
            f"  {'n_steps':>8s}: {'no clip':>9s}  {'clip':>9s}  "
            f"{'rel diff':>10s}"
        )
        for n_steps, r_no, r_clip in results:
            rel = (r_clip / r_no - 1) * 100 if np.isfinite(r_no) else float("nan")
            print(
                f"  {n_steps:8d}: {r_no:8.3f}×  {r_clip:8.3f}×  {rel:+9.1f}%"
            )
        # Interpret
        rel_1 = (results[0][2] / results[0][1] - 1) * 100
        rel_10 = (results[-1][2] / results[-1][1] - 1) * 100
        print(f"\n  clip benefit at 1 step:  {-rel_1:.1f}%")
        print(f"  clip benefit at 10 steps:  {-rel_10:.1f}%")
        if abs(rel_10) < abs(rel_1) * 0.3:
            print(f"  → clip benefit WASHES OUT over time")
        else:
            print(f"  → clip benefit PERSISTS over time")
    assert all(np.isfinite(r) for _, r, _ in results)
    assert all(np.isfinite(r) for _, _, r in results)
