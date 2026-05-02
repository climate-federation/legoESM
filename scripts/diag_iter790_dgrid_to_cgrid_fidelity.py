"""Iter-790 diagnostic: `dgrid_to_cgrid` fidelity for the
solid-body rotation IC at C36.

Iter-789 discovered that `_d2a2c_vect` (measured in iter-782/786)
is NOT in the W2 production path.  The actual W2 D→C operator is
`dgrid_to_cgrid` (src/legoesm/core/operators_cdgrid.py:298).  iter-
790 redoes the iter-782 fidelity test for `dgrid_to_cgrid`.

`dgrid_to_cgrid` is a much simpler operator than `_d2a2c_vect`:
  u_avg   = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])     # (6, n+1, n)
  v_avg_x = 0.5 * (v_d[:, :, :-1] + v_d[:, :, 1:])     # (6, n+1, n)
  sina_u  = sqrt(1 - cosa_u^2)
  u_c     = u_avg * sina_u - v_avg_x * cosa_u
  v_c     = 0.5 * (v_d[:, :-1] + v_d[:, 1:])            # (6, n, n+1)

This is a PHYSICAL edge-midpoint D-grid formulation (used by
FV3EdgeShallowWaterModel), NOT the FV3 corner-D to edge-C
convention with covariant↔contravariant transformation.

Key question: u_d shape is (6, n, n+1) at x-edge; u_avg
averages pairs along j.  But v_d has shape (6, n+1, n) at y-edge.
Averaging `v_d[:, :, :-1]` (shape (6, n+1, n-1)) with
`v_d[:, :, 1:]` (shape (6, n+1, n-1)) gives (6, n+1, n-1) —
that's 2 different shape convention than FV3.  Let me read the
actual code to check the stagger.

Scope: observational only, no source-code change, no new sentinel.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import dgrid_to_cgrid
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, _rotation_winds_geo)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist_to_nearest_cube_vertex_deg(lat_deg, lon_deg):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    min_d = np.inf
    for vlat in CUBE_VERTEX_LATS_R:
        for vlon in CUBE_VERTEX_LONS_R:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return min_d


n = 36
beta = jnp.pi / 4.0

grid = create_cubed_sphere(n)  # use_duogrid=False (W2 production)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=_div_damp_cube(n),
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

state = cosine_bell_cubesphere(grid, cdgrid, beta)
u_c, v_c = dgrid_to_cgrid(state.u_d, state.v_d, cdgrid)
u_c = np.asarray(u_c)
v_c = np.asarray(v_c)

print(f"Iter-790 `dgrid_to_cgrid` fidelity at C36, β=π/4 solid-body rotation")
print(f"u_d shape: {tuple(state.u_d.shape)}, v_d shape: {tuple(state.v_d.shape)}")
print(f"u_c shape: {u_c.shape}, v_c shape: {v_c.shape}")
print()

# The u_c stagger: looking at line 312:
#   u_c = u_avg * sina_u - v_avg_x * cosa_u
# where u_avg is 0.5*(u_d[:, :, :-1] + u_d[:, :, 1:]) with u_d shape
# (6, n, n+1).  So u_avg has shape (6, n, n).  That's CELL CENTRES,
# not C-grid edges.
#
# But comments say "x-face normal velocity".  Let me check by shape.
# If u_d is (6, n, n+1), u_d[:, :, :-1] is (6, n, n), u_d[:, :, 1:] is
# (6, n, n), average is (6, n, n) — CELL CENTRES.
#
# Actually wait, `u_d` in `FV3EdgeShallowWaterState` is edge-midpoint
# (shape (6, n, n+1) — x-edge along j=0..n).  So averaging along j
# gives CELL-CENTER values.  That means u_c here is A CELL-CENTER
# quantity masquerading as "C-grid u" in the comments.  This is
# different from FV3's corner-C convention.
#
# Let me just compare u_c/v_c against the analytical cell-center
# x-normal solid-body rotation winds.

# For edge-midpoint stagger u_c sits at CELL CENTERS, so compute
# analytical at cell centers.
u_e_c = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
# Use geographic (u_east, v_north) at cell center.
u_e_cc, v_n_cc = _rotation_winds_geo(
    grid.lon, grid.lat, grid.radius, beta)
u_e_cc = np.asarray(u_e_cc)
v_n_cc = np.asarray(v_n_cc)

# Project to edge-midpoint D-grid axes (e_i and e_perp).
# From cosine_bell.py lines 152-160:
#   u_d = cos_angle_edge_x * u_east + sin_angle_edge_x * v_north
#   v_d = -sin_angle_edge_y * u_east + cos_angle_edge_y * v_north
# At cell center, we'd have similar with cos_angle/sin_angle at cc.
# But dgrid_to_cgrid's u_c = u_d_avg * sina_u - v_d_avg * cosa_u
# isn't obviously analytical-projectable without understanding the
# non-orthogonality correction.

# Simpler test: the divergence of a solid-body rotation should be
# zero.  Compute divergence of (u_c, v_c) at cell centers (or
# approximate).  Compare to zero.
print(f"max |u_c| = {float(np.max(np.abs(u_c))):.3e}")
print(f"max |v_c| = {float(np.max(np.abs(v_c))):.3e}")

# Interestingly, v_c has shape (6, n, n+1) — j-edge, and u_c has
# shape (6, n+1, n) — i-edge.  Wait that doesn't match what I said
# above.  Let me re-check:
print(f"(re-check) u_c shape: {u_c.shape}, v_c shape: {v_c.shape}")
# n=36, so n+1=37.  u_c should be (6, 37, 36) if it's the x-edge
# (i-edge) and v_c should be (6, 36, 37) if it's the y-edge (j-edge).

# If u_d is (6, n, n+1) at x-edges (n j-slices, n+1 i-positions),
# then u_d[:, :, :-1] is (6, n, n) and u_d[:, :, 1:] is (6, n, n),
# average is (6, n, n).  That's NOT (6, n+1, n).  So the
# `dgrid_to_cgrid` call expects a DIFFERENT shape convention —
# likely u_d at (i, j+1/2) corner?

# Let me just look at the actual shapes.

# The dgrid_to_cgrid docstring says "Works for both 2D (6, n+1, n+1)
# and 3D".  So it expects u_d as (6, n+1, n+1).  But our state.u_d
# is (6, n, n+1) — so these shapes are incompatible.

# This means FV3EdgeShallowWaterModel must have a DIFFERENT
# dgrid_to_cgrid semantic.  Let me check fv3_sw_tendencies to see
# how it actually processes u_d/v_d.
print()
print("NOTE: `dgrid_to_cgrid` expects u_d, v_d at (6, n+1, n+1)")
print("(corner-based D-grid).  FV3EdgeShallowWaterState has u_d at")
print("(6, n, n+1) and v_d at (6, n+1, n) (edge-midpoint D-grid).")
print("These are incompatible shapes.  The actual W2 pipeline must")
print("transform u_d/v_d from edge-midpoint to corner before calling")
print("`dgrid_to_cgrid`, OR use a different operator entirely.")

print()
print("Iter-790 INCONCLUSIVE: shape mismatch reveals operator usage")
print("needs deeper audit.  Iter-791 should trace `fv3_sw_tendencies`")
print("at line 1187 (`u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)`)")
print("to find what u_d, v_d actually look like at that call site.")
