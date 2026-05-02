"""Iter-746b diagnostic: measure `cdgrid.base.angle` variation at
face 4 near-pole cells.

The hyperdiff block in `fv3_sw_tendencies`:
    cos_a = jnp.cos(cdgrid.base.angle)
    sin_a = jnp.sin(cdgrid.base.angle)
    ue_cc = cos_a * u_cc - sin_a * v_cc
    vn_cc = sin_a * u_cc + cos_a * v_cc
    lap_ue = laplacian_compact(ue_cc, cdgrid.base)
    ...
rotates face-local (u_cc, v_cc) to geographic (ue_cc, vn_cc) BEFORE
applying the bilaplacian.  On the polar face (face 4), `angle`
varies rapidly with position because each face-local east direction
points at a different geographic longitude from the pole.  At the
pole, the rotation is degenerate (ill-defined angle).

Hypothesis: the rapid spatial variation of `angle` at face 4 near-
pole injects a high-frequency mode into `ue_cc, vn_cc` that the
bilaplacian amplifies, producing the polar artifact.

Phase 1: print `angle`, `cos(angle)`, `sin(angle)` at face 4 cells
         near the pole.
Phase 2: evaluate `laplacian_compact(ue_cc)` for a known constant
         u_east field — exact answer is zero everywhere.  Any non-
         zero output is INJECTED by the rotation+laplacian pipeline.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter746b_angle_variation.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators import laplacian_compact


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

angle = np.asarray(grid.angle, dtype=np.float64)   # radians, (6, n, n)
angle_deg = np.degrees(angle)
lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi

print("=== Phase 1: angle variation at face 4 near-pole cells ===\n")

print(f"  {'(i,j)':>8}  {'lat':>6}  {'lon':>8}  {'angle_deg':>10}  "
      f"{'cos(a)':>8}  {'sin(a)':>8}")
for i in [17, 18, 19, 20, 21]:
    for j in [15, 16, 17, 18, 19]:
        print(f"  ({i:2d},{j:2d})  "
              f"{lat[4, i, j]:>6.1f}  {lon[4, i, j]:>8.1f}  "
              f"{angle_deg[4, i, j]:>10.2f}  "
              f"{np.cos(angle[4, i, j]):>8.4f}  "
              f"{np.sin(angle[4, i, j]):>8.4f}")

# Measure finite difference of angle across the 3-point x-stencil at
# (19, 17).
print(f"\nAngle 2nd difference across stencils at face 4 (19, 17):")
a_im1 = angle[4, 18, 17]
a_i   = angle[4, 19, 17]
a_ip1 = angle[4, 20, 17]
print(f"  x-stencil: angle[18,17]={np.degrees(a_im1):.2f}°, "
      f"angle[19,17]={np.degrees(a_i):.2f}°, "
      f"angle[20,17]={np.degrees(a_ip1):.2f}°")
print(f"  Δ_x angle = {np.degrees(a_ip1 - a_im1):.2f}°")
print(f"  2nd diff  = {np.degrees(a_im1 - 2*a_i + a_ip1):.4f}°")

a_jm1 = angle[4, 19, 16]
a_jp1 = angle[4, 19, 18]
print(f"\n  y-stencil: angle[19,16]={np.degrees(a_jm1):.2f}°, "
      f"angle[19,17]={np.degrees(a_i):.2f}°, "
      f"angle[19,18]={np.degrees(a_jp1):.2f}°")
print(f"  Δ_y angle = {np.degrees(a_jp1 - a_jm1):.2f}°")
print(f"  2nd diff  = {np.degrees(a_jm1 - 2*a_i + a_jp1):.4f}°")

# Compare to equatorial face interior.
print(f"\nEquatorial face 0, cell (18, 18) for comparison:")
a_im1 = angle[0, 17, 18]
a_i   = angle[0, 18, 18]
a_ip1 = angle[0, 19, 18]
print(f"  x-stencil Δ_x angle: {np.degrees(a_ip1 - a_im1):.4f}°")
print(f"  x-stencil 2nd diff:  {np.degrees(a_im1 - 2*a_i + a_ip1):.6f}°")

print()
print("=== Phase 2: Does rotate→laplacian inject an artifact from a ")
print("            CONSTANT u_east field? ===\n")

# Construct constant u_east = 10 m/s, v_north = 0 everywhere.
u_east_const = 10.0 * np.ones_like(lat)
v_north_const = 0.0 * np.ones_like(lat)

# Rotate into face-local (u_cc, v_cc).
cos_a = np.cos(angle)
sin_a = np.sin(angle)
u_cc = cos_a * u_east_const + sin_a * v_north_const
v_cc = -sin_a * u_east_const + cos_a * v_north_const

# Now apply the hyperdiff pipeline: rotate back to (ue, vn), laplacian,
# bilaplacian.
ue_cc = cos_a * u_cc - sin_a * v_cc  # should return 10 m/s
vn_cc = sin_a * u_cc + cos_a * v_cc  # should return 0 m/s

ue_cc_jx = jnp.asarray(ue_cc)
vn_cc_jx = jnp.asarray(vn_cc)

print(f"After round-trip face-local → geographic:")
print(f"  max|ue_cc - 10|   = {np.abs(np.asarray(ue_cc) - 10.0).max():.3e}")
print(f"  max|vn_cc - 0|    = {np.abs(np.asarray(vn_cc) - 0.0).max():.3e}")
print()

lap_ue = np.asarray(laplacian_compact(ue_cc_jx, grid))
lap_vn = np.asarray(laplacian_compact(vn_cc_jx, grid))
bilap_ue = np.asarray(laplacian_compact(jnp.asarray(lap_ue), grid))
bilap_vn = np.asarray(laplacian_compact(jnp.asarray(lap_vn), grid))

print(f"Laplacian of a CONSTANT geographic field should be 0.")
print(f"  max|lap(ue_cc)|   = {np.abs(lap_ue).max():.3e}  (expected 0)")
print(f"  max|lap(vn_cc)|   = {np.abs(lap_vn).max():.3e}  (expected 0)")
print(f"  max|bilap(ue_cc)| = {np.abs(bilap_ue).max():.3e}  (expected 0)")
print(f"  max|bilap(vn_cc)| = {np.abs(bilap_vn).max():.3e}  (expected 0)")

# Per-face analysis.
print(f"\nPer-face max|lap(ue_cc)|:")
for f in range(6):
    val = np.abs(lap_ue[f]).max()
    argmax = np.argmax(np.abs(lap_ue[f]))
    i, j = np.unravel_index(argmax, lap_ue[f].shape)
    print(f"  face{f}: max={val:.3e}  at (i={i},j={j}), "
          f"lat={lat[f,i,j]:.1f}°, lon={lon[f,i,j]:.1f}°")

# Rotate bilap_ue/bilap_vn back to face-local (what hyperdiff actually
# applies).
bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn

# Compute the hyperdiff contribution at each cell.
# du_cc = du_cc - hyperdiff_coeff * bilap_u_local
# For W2 where u_east is close to constant (within cos(lat)), this
# contribution represents the SPURIOUS cell-centre tendency injected
# by hyperdiff from a field that should have no tendency.
from scripts.run_atmosphere_test_matrix import _hyperdiff_cube
hyp_coeff = _hyperdiff_cube(n)
print(f"\nHyperdiff coeff at C36: {hyp_coeff:.3e}")
print(f"Spurious du_cc = -hyperdiff_coeff * bilap_u_local:")
spurious_du = hyp_coeff * np.abs(bilap_u_local)
spurious_dv = hyp_coeff * np.abs(bilap_v_local)
print(f"  Global max|du|: {spurious_du.max():.3e} m/s/s")
print(f"  Global max|dv|: {spurious_dv.max():.3e} m/s/s")
print(f"  Over 1 day (86400s): max|Δu| = {spurious_du.max()*86400:.3e} m/s")
print(f"                       max|Δv| = {spurious_dv.max()*86400:.3e} m/s")

# Face 4/5 Linf.
print(f"\nFace 4/5 |bilap_u_local|:  {max(np.abs(bilap_u_local[4]).max(), np.abs(bilap_u_local[5]).max()):.3e}")
print(f"Face 0-3 |bilap_u_local|:  {max(np.abs(bilap_u_local[f]).max() for f in range(4)):.3e}")

# Interpretation.
print()
print("Interpretation: if 'max|Δv| over 1 day' is comparable to the")
print("iter-745 polar peak of 0.303 m/s, the rotate→laplacian→rotate")
print("pipeline is DIRECTLY creating the artifact from a constant")
print("u_east field.  Smaller values would mean the error arises only")
print("when the underlying u_east has its OWN non-trivial structure.")
