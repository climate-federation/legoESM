"""FV3_3D iter 517: smooth IC vs random IC — is the residual
edge bias intrinsic to the dycore, or driven by IC noise?

iters 509-516 used uniform-random u/v initial fields, which
contain energy at all wavenumbers including the grid scale.
Grid-scale noise is what corner stencils amplify most.

Real atmospheric initial conditions are smooth (spectral
content peaks at large scales).  If a smooth IC produces a
much smaller e/i ratio, then the 2× residual was a stress-
test, not a real concern for atmospheric runs.

Tests
-----

1. ``test_smooth_vs_random_ic_at_10_steps`` — initialize with
   a smooth solid-body-like u, v pattern (sin/cos of face
   position).  Compare 10-step duogrid e/i to random-IC result.
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


def _build_state_random(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    return grid, hc, tm, u_p, v_p, nlev


def _build_state_smooth(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    # Smooth u, v: solid-body-rotation-like.  Use face-relative
    # smooth pattern.
    rng = np.random.default_rng(seed=seed)
    i_idx, j_idx = np.meshgrid(
        np.arange(n) / max(n - 1, 1),
        np.arange(n) / max(n - 1, 1),
        indexing="ij",
    )
    # smooth field varying on (i, j) ∈ [0, 1]^2, single-mode sinusoid
    base = 3.0 * np.sin(np.pi * i_idx) * np.cos(np.pi * j_idx)
    u_p = np.zeros((6, n, n, nlev))
    v_p = np.zeros((6, n, n, nlev))
    for f in range(6):
        for k in range(nlev):
            u_p[f, :, :, k] = base
            v_p[f, :, :, k] = -base  # divergence-free-ish
    return grid, hc, tm, u_p, v_p, nlev


def _make_state(u_p, v_p, n, nlev):
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    return NonHydrostaticState(
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


def _run(ic_builder, seed):
    n = 8
    grid, hc, tm, u_p, v_p, nlev = ic_builder(n, seed)
    state = _make_state(u_p, v_p, n, nlev)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    with monotone_halo_clip_context(slack=0.5):
        m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        s = state
        for _ in range(10):
            s = m.step(s, dt=10.0)
    e, i = _edge_and_interior_std(s.theta_prime.data)
    return e / i if i > 1e-30 else float("nan")


def test_smooth_vs_random_ic_at_10_steps(capsys):
    seed = 517
    r_random = _run(_build_state_random, seed)
    r_smooth = _run(_build_state_smooth, seed)
    with capsys.disabled():
        print(
            f"\n[iter-517 smooth vs random IC @ 10 steps "
            f"(seed=517, C8, min-edge + clip)]"
        )
        print(f"  random IC (u,v ∈ [-3, 3]):   e/i = {r_random:.3f}×")
        print(f"  smooth IC (sinusoidal u,v): e/i = {r_smooth:.3f}×")
        if np.isfinite(r_smooth) and np.isfinite(r_random):
            print(
                f"\n  ratio random/smooth: {r_random/r_smooth:.2f}× — "
                f"{(r_random/r_smooth - 1)*100:+.0f}% noise penalty"
            )
            if r_smooth < 1.2:
                print(
                    f"  smooth e/i < 1.2 → real-IC runs are nearly "
                    f"artifact-free"
                )
            else:
                print(
                    f"  smooth e/i >= 1.2 → artifact also present in "
                    f"smooth runs"
                )
    assert np.isfinite(r_random) and np.isfinite(r_smooth)
