"""Iter-746 diagnostic: quantify dx variation at face 4 near-pole
cells, and test whether a variable-spacing compact Laplacian reduces
the W2 polar hyperdiff residual.

Iter-745 showed that hyperdiff adds +0.087 m/s (28.7%) to the W2
polar peak at face 4 (i=19, j=17).  Iter-745b ruled out halo
rotation.  Iter-745 ordered work item 4 (promoted): test whether the
`laplacian_compact` formula
    d²f/dx² ≈ (f[i+1] - 2f[i] + f[i-1]) / (dx[i,j]/2)²
introduces an O(Δdx/dx) error at face-4 near-pole cells where dx
varies significantly across the 3-point stencil.

Phase 1: measure dx[i,j] variation for face 4 stencils centred on
         the known polar peak cell (19, 17) and its neighbours.
Phase 2: implement a non-uniform-spacing compact Laplacian and
         compare hyperdiff output at the polar cells against the
         current uniform-spacing formula.

No source-code change committed unless Phase 2 shows a meaningful
reduction in the hyperdiff contribution AT the polar peak.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter746_polar_dx_variation.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

dx = np.asarray(grid.dx, dtype=np.float64)   # (6, n, n)
dy = np.asarray(grid.dy, dtype=np.float64)
lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi

print("=== Phase 1: dx variation at face 4 near-pole cells ===\n")

# The iter-745 polar peak is at face 4 (i=19, j=17) and face 5
# (i=19, j=18).  Let's look at the 3-point x-stencil at that cell:
#   padded[:, i+2, j+1] - 2*padded[:, i+1, j+1] + padded[:, i, j+1]
# (in padded coords).  The local dx is dx[face, i, j].  For a
# uniform-spacing formula to be accurate, dx[face, i-1, j], dx[face,
# i, j], dx[face, i+1, j] should all be ~equal.

print("Face 4 polar region (near cell (19, 17)):\n")
print(f"  {'(i,j)':>8}  {'lat':>7}  {'lon':>8}  "
      f"{'dx':>10}  {'dy':>10}  {'dx/dy':>7}")
for i in [17, 18, 19, 20, 21]:
    for j in [15, 16, 17, 18, 19]:
        print(f"  ({i:2d},{j:2d})  "
              f"{lat[4, i, j]:>7.2f}  {lon[4, i, j]:>8.2f}  "
              f"{dx[4, i, j]:>10.2f}  {dy[4, i, j]:>10.2f}  "
              f"{dx[4, i, j] / dy[4, i, j]:>7.3f}")

print("\ndx variation across the 3-point x-stencil at (19, 17):")
dx_im1 = dx[4, 18, 17]
dx_i   = dx[4, 19, 17]
dx_ip1 = dx[4, 20, 17]
print(f"  dx[18,17] = {dx_im1:.2f}")
print(f"  dx[19,17] = {dx_i:.2f}")
print(f"  dx[20,17] = {dx_ip1:.2f}")
print(f"  ratio (im1/i)   = {dx_im1/dx_i:.4f}")
print(f"  ratio (ip1/i)   = {dx_ip1/dx_i:.4f}")
delta = (dx_im1 - 2*dx_i + dx_ip1) / dx_i
print(f"  relative 2nd diff ({dx_im1 - 2*dx_i + dx_ip1:.3f} / {dx_i:.2f}) = "
      f"{delta:.4f}")

# Compare to an equatorial face cell far from any distorted metric.
print("\nEquatorial face 0, interior cell (18, 18) for comparison:")
dx_im1_eq = dx[0, 17, 18]
dx_i_eq   = dx[0, 18, 18]
dx_ip1_eq = dx[0, 19, 18]
print(f"  dx[17,18] = {dx_im1_eq:.2f}")
print(f"  dx[18,18] = {dx_i_eq:.2f}")
print(f"  dx[19,18] = {dx_ip1_eq:.2f}")
delta_eq = (dx_im1_eq - 2*dx_i_eq + dx_ip1_eq) / dx_i_eq
print(f"  relative 2nd diff = {delta_eq:.4f}")

# Same for face 4 dy at (19, 17):
print("\ndy variation across the 3-point y-stencil at face 4 (19, 17):")
dy_jm1 = dy[4, 19, 16]
dy_j   = dy[4, 19, 17]
dy_jp1 = dy[4, 19, 18]
print(f"  dy[19,16] = {dy_jm1:.2f}")
print(f"  dy[19,17] = {dy_j:.2f}")
print(f"  dy[19,18] = {dy_jp1:.2f}")
delta_y = (dy_jm1 - 2*dy_j + dy_jp1) / dy_j
print(f"  relative 2nd diff = {delta_y:.4f}")

print()
print("Interpretation: if the relative 2nd diff of dx is >0.05 at the")
print("polar peak cell, the uniform-spacing laplacian_compact formula")
print("has a comparable-magnitude error relative to the 2nd diff of")
print("the field itself — a plausible mechanism for the hyperdiff")
print("polar residual.")
print()

print("=== Phase 2: non-uniform vs uniform compact Laplacian on a ")
print("             smooth analytic field at face 4 ===\n")

# Build a smooth analytic field that has a KNOWN Laplacian — e.g.,
# f = sin(lat) (a zonal mode-0 pattern).  The Laplacian of sin(lat) on
# a sphere is -2 sin(lat) / a² (Earth's radius a).  At lat=86°,
# sin(86°)=0.998, so ∇²f_exact = -2 * 0.998 / a².
radius = float(grid.radius)
lat_face = np.asarray(grid.lat, dtype=np.float64)  # radians
f_analytic = np.sin(lat_face)  # (6, n, n)
f_analytic_jx = jnp.asarray(f_analytic)

# True Laplacian on a sphere for sin(lat): d/dθ(cos(θ) dsin(θ)/dθ) /
# (cos(θ) * a^2) = -sin(θ) * 2 / a^2.  Wait that's not right; let me
# redo: let f = sin(lat).  Then ∇²f = (1/cos(lat)) d/dlat (cos(lat) *
# df/dlat) / a^2 = (1/cos(lat)) d/dlat (cos²(lat)) / a^2.  d/dlat
# cos²(lat) = -2 cos(lat) sin(lat), so ∇²f = -2 sin(lat) / a^2.
lap_exact = -2.0 * np.sin(lat_face) / radius**2

# Uniform-spacing compact Laplacian (current production formula):
from legoesm.core.operators import laplacian_compact
lap_uniform = np.asarray(laplacian_compact(f_analytic_jx, grid))

# Non-uniform variable-spacing compact Laplacian:
# For cell i with neighbours at i-1 and i+1, if cell centres are at
# spacing h_minus = dx_half[i-1] + dx_half[i] (approximately) and
# h_plus = dx_half[i] + dx_half[i+1], then
#   d²f/dx² ≈ 2 * [(f[i+1] - f[i])/h_plus - (f[i] - f[i-1])/h_minus]
#                 / (h_plus + h_minus)
# Using grid.dx[i,j] as the FULL cell width (so h_plus ≈ (dx[i]+dx[i+1])/2):
from legoesm.grids import halo as halo_mod

dg = getattr(grid, 'duogrid', None)
offsets = None if dg is not None else grid.halo_interp_offsets
f_padded = np.asarray(halo_mod.pad_halo(
    f_analytic_jx, interp_offsets=offsets, duogrid=dg))
dx_padded = np.asarray(halo_mod.pad_halo(
    jnp.asarray(dx), interp_offsets=offsets, duogrid=dg))
dy_padded = np.asarray(halo_mod.pad_halo(
    jnp.asarray(dy), interp_offsets=offsets, duogrid=dg))

dx_im1 = dx_padded[:, :-2, 1:-1]    # dx at i-1
dx_i   = dx_padded[:, 1:-1, 1:-1]   # dx at i
dx_ip1 = dx_padded[:, 2:, 1:-1]     # dx at i+1
# Cell-to-cell spacing (centre-to-centre).
h_minus_x = 0.5 * (dx_im1 + dx_i)
h_plus_x = 0.5 * (dx_i + dx_ip1)

f_im1 = f_padded[:, :-2, 1:-1]
f_i   = f_padded[:, 1:-1, 1:-1]
f_ip1 = f_padded[:, 2:, 1:-1]

d2f_dx2_nu = 2.0 * ((f_ip1 - f_i) / h_plus_x
                    - (f_i - f_im1) / h_minus_x) / (h_plus_x + h_minus_x)

dy_jm1 = dy_padded[:, 1:-1, :-2]
dy_j   = dy_padded[:, 1:-1, 1:-1]
dy_jp1 = dy_padded[:, 1:-1, 2:]
h_minus_y = 0.5 * (dy_jm1 + dy_j)
h_plus_y = 0.5 * (dy_j + dy_jp1)

f_jm1 = f_padded[:, 1:-1, :-2]
f_jp1 = f_padded[:, 1:-1, 2:]

d2f_dy2_nu = 2.0 * ((f_jp1 - f_i) / h_plus_y
                    - (f_i - f_jm1) / h_minus_y) / (h_plus_y + h_minus_y)

lap_nonuniform = d2f_dx2_nu + d2f_dy2_nu

# Compare on the polar peak cell AND on a reference equatorial cell.
print(f"Field f = sin(lat).  Exact laplacian -2 sin(lat) / a² at lat=86°")
print(f"                                                = {-2*np.sin(np.radians(86))/radius**2:.4e}")
print()
print(f"{'Cell':>18}  {'lat':>6}  {'exact':>12}  {'uniform':>12}  "
      f"{'non-unif':>12}  {'err_unif':>11}  {'err_nu':>11}")
for face, i, j in [(4, 19, 17), (4, 18, 18), (4, 0, 0), (0, 18, 18),
                   (0, 2, 2)]:
    ex = lap_exact[face, i, j]
    un = lap_uniform[face, i, j]
    nu = lap_nonuniform[face, i, j]
    err_un = (un - ex) / abs(ex) if abs(ex) > 1e-20 else np.nan
    err_nu = (nu - ex) / abs(ex) if abs(ex) > 1e-20 else np.nan
    print(f"  face{face},({i:2d},{j:2d})  {lat[face,i,j]:>6.1f}  "
          f"{ex:>12.4e}  {un:>12.4e}  {nu:>12.4e}  "
          f"{err_un:>10.2%}  {err_nu:>10.2%}")

# Global norms.
err_unif_all = np.abs(lap_uniform - lap_exact)
err_nu_all = np.abs(lap_nonuniform - lap_exact)
face4_mask = np.zeros_like(err_unif_all)
face4_mask[4] = 1.0
face4_mask[5] = 1.0
print(f"\nGlobal Linf  (uniform):   {err_unif_all.max():.4e}")
print(f"Global Linf  (non-unif):  {err_nu_all.max():.4e}")
print(f"Face-4/5 Linf (uniform): {(err_unif_all * face4_mask).max():.4e}")
print(f"Face-4/5 Linf (non-unif): {(err_nu_all * face4_mask).max():.4e}")
print(f"Face-0/1/2/3 Linf (uniform):   "
      f"{(err_unif_all * (1 - face4_mask)).max():.4e}")
print(f"Face-0/1/2/3 Linf (non-unif):  "
      f"{(err_nu_all * (1 - face4_mask)).max():.4e}")

# Location of max error in the uniform case.
face_max = int(np.argmax(err_unif_all.reshape(6, -1).max(axis=1)))
flat_max = int(np.argmax(err_unif_all[face_max]))
i_max, j_max = np.unravel_index(flat_max, err_unif_all[face_max].shape)
print(f"\nUniform max-error location: face{face_max}, "
      f"(i={i_max}, j={j_max}), lat={lat[face_max,i_max,j_max]:.2f}°, "
      f"lon={lon[face_max,i_max,j_max]:.2f}°")
