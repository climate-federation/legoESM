#!/usr/bin/env python
"""Test cubic halo interpolation + edge blend for FV cubed-sphere.

Patches _interp_strip in halo.py to use cubic Catmull-Rom,
adds edge blending, and tests on Williamson TC2.
"""

import sys
sys.path.insert(0, "tests")

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids import halo as halo_mod
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from test_cases.williamson import williamson_test2, compute_error_norms

# ---- Patch 1: Cubic halo interpolation ----

_orig_interp_strip = halo_mod._interp_strip

def _interp_strip_cubic(strip, offsets_1d):
    """Cubic Catmull-Rom interpolation, linear fallback at strip edges."""
    n = strip.shape[0]
    idx = jnp.arange(n, dtype=offsets_1d.dtype) + offsets_1d
    idx = jnp.clip(idx, 0.0, n - 1.0)
    lo = jnp.floor(idx).astype(jnp.int32)
    t = idx - lo.astype(offsets_1d.dtype)
    t = jnp.clip(t, 0.0, 1.0)

    # Linear (fallback)
    lo_lin = jnp.clip(lo, 0, n - 2)
    val_linear = (1.0 - t) * strip[lo_lin] + t * strip[lo_lin + 1]

    # Cubic Catmull-Rom
    lo_cub = jnp.clip(lo, 1, n - 3)
    p0 = strip[lo_cub - 1]
    p1 = strip[lo_cub]
    p2 = strip[lo_cub + 1]
    p3 = strip[lo_cub + 2]
    t2 = t * t
    t3 = t2 * t
    val_cubic = 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    can_cubic = (lo >= 1) & (lo <= n - 3)
    return jnp.where(can_cubic, val_cubic, val_linear)


# ---- Patch 2: Edge blending ----

from legoesm.grids.halo import pad_halo, pad_halo_vector

def face_boundary_weight(n, depth=2, strength=0.25):
    """Smooth weight: strength at boundary, 0 in interior."""
    w = jnp.zeros((n, n))
    for d in range(depth):
        alpha = strength * (1.0 - d / depth)
        row = jnp.full(n, alpha)
        w = w.at[d, :].set(jnp.maximum(w[d, :], row))
        w = w.at[-(d+1), :].set(jnp.maximum(w[-(d+1), :], row))
        w = w.at[:, d].set(jnp.maximum(w[:, d], row))
        w = w.at[:, -(d+1)].set(jnp.maximum(w[:, -(d+1)], row))
    return w[None]  # (1, n, n) broadcastable to (6, n, n)

def edge_blend_scalar(q, grid, weight):
    """Apply localized Laplacian smoothing near face boundaries."""
    q_pad = pad_halo(q, interp_offsets=grid.halo_interp_offsets)
    avg = (q_pad[:, 2:, 1:-1] + q_pad[:, :-2, 1:-1] +
           q_pad[:, 1:-1, 2:] + q_pad[:, 1:-1, :-2]) / 4.0
    return q + weight * (avg - q)

def edge_blend_vector(u, v, grid, weight):
    """Edge blend for velocity (using proper vector halo)."""
    u_pad, v_pad = pad_halo_vector(
        u, v, grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )
    avg_u = (u_pad[:, 2:, 1:-1] + u_pad[:, :-2, 1:-1] +
             u_pad[:, 1:-1, 2:] + u_pad[:, 1:-1, :-2]) / 4.0
    avg_v = (v_pad[:, 2:, 1:-1] + v_pad[:, :-2, 1:-1] +
             v_pad[:, 1:-1, 2:] + v_pad[:, 1:-1, :-2]) / 4.0
    return u + weight * (avg_u - u), v + weight * (avg_v - v)


# ---- Run tests ----

from legoesm.atmosphere.dynamics.shallow_water_fv import (
    FVShallowWaterModel,
    FVShallowWaterConfig,
    fv_shallow_water_tendencies,
)
from legoesm.core.state import ShallowWaterState

N = 24
grid = create_cubed_sphere(N)
ic = williamson_test2(grid)
ref = ic
dt = 450.0
n_steps = int(5 * 86400 / dt)
nu2, nu4 = default_div_damp_coeffs(grid, dt=dt)
dx_min = float(jnp.min(grid.dx / 2.0))

def run_test(name, use_cubic=False, edge_blend_depth=0, edge_blend_str=0.0, hyperdiff=0.0):
    """Run TC2 for 5 days and report errors."""
    # Patch halo interpolation
    if use_cubic:
        halo_mod._interp_strip = _interp_strip_cubic
    else:
        halo_mod._interp_strip = _orig_interp_strip

    config = FVShallowWaterConfig(
        div_damp_2=nu2, div_damp_4=nu4,
        hyperdiff_coeff=hyperdiff,
    )
    model = FVShallowWaterModel(grid, config)

    # Precompute edge blend weight
    if edge_blend_depth > 0 and edge_blend_str > 0:
        eb_w = face_boundary_weight(N, edge_blend_depth, edge_blend_str)
    else:
        eb_w = None

    state = ic
    for i in range(n_steps):
        state = model.step(state, dt)
        # Apply edge blend after step
        if eb_w is not None:
            h_new = edge_blend_scalar(state.h.data, grid, eb_w)
            u_new, v_new = edge_blend_vector(state.u.data, state.v.data, grid, eb_w)
            state = ShallowWaterState(
                h=state.h.replace(data=h_new),
                u=state.u.replace(data=u_new),
                v=state.v.replace(data=v_new),
                h_s=state.h_s,
            )

        if jnp.any(~jnp.isfinite(state.h.data)):
            print(f"  {name}: BLEW UP at step {i+1}")
            halo_mod._interp_strip = _orig_interp_strip
            return

    norms = compute_error_norms(state, ref, grid)
    u_err = float(jnp.max(jnp.abs(state.u.data - ic.u.data)))
    v_err = float(jnp.max(jnp.abs(state.v.data - ic.v.data)))
    print(f"  {name}: L2={norms['l2']:.4e}  Linf={norms['linf']:.4e}  |du|={u_err:.1f}  |dv|={v_err:.1f}")

    # Restore
    halo_mod._interp_strip = _orig_interp_strip


print(f"Williamson TC2, C{N}, dt={dt}s, {n_steps} steps (5 days)")
print(f"nu2={nu2:.3e}, nu4={nu4:.3e}, dx_min={dx_min:.0f}")
print()

hyp = 0.05 * dx_min**4 / dt

# Test configurations
run_test("1. baseline (div damp only)")
run_test("2. cubic halo", use_cubic=True)
run_test("3. edge blend d=2 s=0.25", edge_blend_depth=2, edge_blend_str=0.25)
run_test("4. hyperdiff 0.05", hyperdiff=hyp)
run_test("5. cubic + edge blend", use_cubic=True, edge_blend_depth=2, edge_blend_str=0.25)
run_test("6. cubic + hyperdiff", use_cubic=True, hyperdiff=hyp)
run_test("7. cubic + edge + hyp", use_cubic=True, edge_blend_depth=2, edge_blend_str=0.25, hyperdiff=hyp)
run_test("8. edge + hyp", edge_blend_depth=2, edge_blend_str=0.25, hyperdiff=hyp)

print("\nDone.")
