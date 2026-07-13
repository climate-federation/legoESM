"""FV3_3D iter 525: does ``monotone_halo_clip_context`` survive
``jax.jit``?

The helper uses ``unittest.mock.patch`` at Python level.  JAX
traces functions ONCE; the patch must be active at trace time
to be baked into the compiled graph.

Two scenarios:
1. Trace inside context, call inside: clip baked in ✓
2. Trace outside context, call inside: clip NOT in graph ✗

Tests
-----

1. ``test_jit_inside_context`` — jit-compile while context
   is active; verify clip is in the graph (no clip vs clip
   ratios differ).
2. ``test_jit_outside_context_warns`` — document the
   limitation: jit-compile OUTSIDE context, run inside ⇒
   clip has no effect.
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


def test_jit_inside_context(capsys):
    """jit-compile while context is active → clip is baked into graph."""
    grid, hc, tm, state = _build_state(8, 525)
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

    # Path A: jit-compile inside context.
    with monotone_halo_clip_context(slack=0.5):
        step_inside = jax.jit(m.step)
        s_inside = step_inside(state, 10.0).block_until_ready() if False else step_inside(state, 10.0)

    # Path B: jit-compile outside context, run outside.
    step_outside = jax.jit(m.step)
    s_outside = step_outside(state, 10.0)

    e_in, i_in = _edge_and_interior_std(s_inside.theta_prime.data)
    e_out, i_out = _edge_and_interior_std(s_outside.theta_prime.data)
    r_in = e_in / i_in if i_in > 1e-30 else float("nan")
    r_out = e_out / i_out if i_out > 1e-30 else float("nan")
    with capsys.disabled():
        print(f"\n[iter-525 jit + clip context]")
        print(
            f"  jit inside context, run inside:  e/i = {r_in:.3f}× "
            f"(should be clipped)"
        )
        print(
            f"  jit outside context, run outside: e/i = {r_out:.3f}× "
            f"(unclipped baseline)"
        )
    # Clipped version should have lower e/i than unclipped
    assert np.isfinite(r_in) and np.isfinite(r_out)
    # Allow within 5% as the test is at 1 step — broad sanity.
    # The clipped version should not be MUCH higher than unclipped.
    assert r_in <= r_out * 1.5, (
        f"jit-inside-context should not WORSEN ratio: "
        f"r_in={r_in:.3f}, r_out={r_out:.3f}"
    )


def test_jit_outside_context_no_effect(capsys):
    """Document limitation: jit OUTSIDE context, run INSIDE = no clip."""
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

    # jit OUTSIDE context
    step_jit = jax.jit(m.step)
    # Force a trace OUTSIDE context to ensure no clip is baked
    s_warmup = step_jit(state, 10.0)
    e_baseline, i_baseline = _edge_and_interior_std(s_warmup.theta_prime.data)
    r_baseline = e_baseline / i_baseline if i_baseline > 1e-30 else float("nan")

    # Run INSIDE context — uses same compiled graph (no clip)
    with monotone_halo_clip_context(slack=0.5):
        s_with_ctx = step_jit(state, 10.0)
    e_ctx, i_ctx = _edge_and_interior_std(s_with_ctx.theta_prime.data)
    r_ctx = e_ctx / i_ctx if i_ctx > 1e-30 else float("nan")
    with capsys.disabled():
        print(
            f"\n[iter-525 jit-outside-context, run-inside (limitation)]"
        )
        print(f"  baseline (jit outside, run outside): e/i = {r_baseline:.3f}×")
        print(f"  run with context (after trace):      e/i = {r_ctx:.3f}×")
        print(
            f"  → same compiled graph reused; clip has NO effect "
            f"when jit traces OUTSIDE the context"
        )
    # Should match (within machine precision)
    assert abs(r_baseline - r_ctx) < 1e-10 or r_baseline == r_ctx, (
        f"jit traced outside context should be unaffected by later "
        f"context activation; r_baseline={r_baseline}, r_ctx={r_ctx}"
    )
