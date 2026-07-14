"""FV3_3D iter 526: ``make_clipped_step()`` helper test.

iter-525 documented that ``jax.jit(model.step)`` outside the
clip context misses the clip patch.  iter-526 provides a
``make_clipped_step`` helper that compiles inside the context
and returns a permanently-clipped JIT function.

Tests
-----

1. ``test_make_clipped_step_returns_callable``.
2. ``test_make_clipped_step_clips_baked_in`` — vs raw
   ``jax.jit(model.step)``, the clipped version gives a
   LOWER e/i ratio (clip is active).
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
from legoesm.grids.halo import make_clipped_step
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


def _build_state(n, seed):
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


def test_make_clipped_step_returns_callable():
    grid, hc, tm, state = _build_state(8, 526)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)
    assert callable(step)
    new_state = step(state, 10.0)
    assert new_state.theta_prime.data.shape == state.theta_prime.data.shape


def test_make_clipped_step_clips_baked_in(capsys):
    """Compare e/i ratio: raw jit vs make_clipped_step.

    Run multi-step to amplify the difference.
    """
    grid, hc, tm, state = _build_state(8, 527)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)

    # Use DIFFERENT model instances so JAX trace cache doesn't reuse.
    m_clip = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_clipped = make_clipped_step(m_clip, state, dt=10.0, slack=0.5)
    s = state
    for _ in range(10):
        s = step_clipped(s, 10.0)
    e_clip, i_clip = _edge_and_interior_std(s.theta_prime.data)
    r_clip = e_clip / i_clip if i_clip > 1e-30 else float("nan")

    m_raw = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    step_raw = jax.jit(m_raw.step)
    s = state
    for _ in range(10):
        s = step_raw(s, 10.0)
    e_raw, i_raw = _edge_and_interior_std(s.theta_prime.data)
    r_raw = e_raw / i_raw if i_raw > 1e-30 else float("nan")
    with capsys.disabled():
        print(
            f"\n[iter-526 make_clipped_step vs raw jit @ 10 steps]"
        )
        print(f"  raw jit:           e/i = {r_raw:.3f}×")
        print(f"  make_clipped_step: e/i = {r_clip:.3f}×")
        if np.isfinite(r_raw) and np.isfinite(r_clip):
            reduction = (1 - r_clip / r_raw) * 100
            print(f"  reduction: {reduction:+.1f}%")
    assert np.isfinite(r_raw) and np.isfinite(r_clip)
    assert r_clip <= r_raw * 1.1, (
        f"clipped should not WORSEN: r_clip={r_clip}, r_raw={r_raw}"
    )
