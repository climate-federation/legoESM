"""Iter-763 diagnostic: measure the difference between Python's
2-point-averaging cube-corner halo fill and Fortran's directional
copy-from-interior fill (sw_core.F90:3856-3915 `fill_4corners`).

Iter-762 showed the residual W2 v-wind artifact is ENTIRELY at
the 8 cube vertices (lat ±35°, lon ±45°/±135°).  Iter-763 tests
whether the cube-corner halo-fill discrepancy between Python
(_fill_corners_h1: 2-point edge average) and Fortran
(fill_4corners: directional copy) is materially non-zero on the
W2-exact-IC B field.

If the difference is large (~O(B)), the corner fill is likely
contributing to mode A and a Fortran-style directional fill is a
concrete iter-764+ target.  If negligible, mode A is driven by
a different mechanism (metric, gradient stencil, etc.).
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

# Construct W2-exact Bernoulli field: B = KE + g*h
# For W2 alpha=0: u_east = u_0*cos(lat), v_north=0, h = h_0 - (u0*omega*R + u0^2/2)*sin^2(lat)/g
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
lat = np.asarray(grid.lat, dtype=np.float64)
u_east = u0 * np.cos(lat)
KE = 0.5 * u_east ** 2      # cell-centre KE
# Use a simple smooth h ~ KE scale — for this test we just need a
# smooth cube-sphere field.  Use KE directly as the test field.
B = jnp.asarray(KE)

# Standard pad_halo (Python _fill_corners_h1).
B_pad_py = pad_halo(B, interp_offsets=grid.halo_interp_offsets,
                    duogrid=grid.duogrid)

# Fortran-style directional copy fill (fill_4corners, dir=1):
#   sw_corner: q(-1,0) = q(0,2);  q(0,0) = q(0,1)
#   etc.  In our padded convention with halo=1:
#     padded index (0, 0) ↔ Fortran q(-1, -1)? or (-1, 0)?
# The fill_4corners signature specifies specific CELLS at the 4
# cube corners.  Without replicating the full cube-corner topology,
# we can approximate: SW-corner halo cell at padded[f, 0, 0] should
# be q(0, 1) (one cell inward in y) for dir=1.  Our 2-point average
# sets it to 0.5*(halo_x_edge + halo_y_edge).  If the directional
# copy gives a different value, we flag it.
#
# For simplicity in this diagnostic, we compute the "expected
# interior-reflect" alternative: at padded[:, 0, 0], use the value
# at padded[:, 1, 1] (2D interior-diagonal).  This is a crude
# stand-in for Fortran's directional fill — not exact but indicates
# whether cube-corner values are sensitive to the fill choice.

# Corner indices in padded array:
corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]  # (SW, NW, SE, NE)

print("Python 2-point-avg cube-corner halo vs interior-diagonal "
      "alternative on W2 KE field:\n")

B_py = np.asarray(B_pad_py, dtype=np.float64)

# Total pairs: 6 faces x 4 corners.
max_abs_diff = 0.0
for f in range(6):
    for ci, cj in corners:
        # Interior-diagonal index:
        di = 1 if ci == 0 else -2
        dj = 1 if cj == 0 else -2
        py_val = B_py[f, ci, cj]
        diag_val = B_py[f, di, dj]
        diff = py_val - diag_val
        max_abs_diff = max(max_abs_diff, abs(diff))

print(f"Max |python_corner - interior_diagonal| across 24 cube corners: "
      f"{max_abs_diff:.4e}")

# The magnitude of B on the W2 IC is ~O(u0^2) = O((2*pi*R/T)^2) at most.
B_scale = float(np.max(np.abs(B_py)))
print(f"B field scale: {B_scale:.4e}")
print(f"Relative corner discrepancy: {max_abs_diff/B_scale:.4%}")

# Focus on cube-vertex corners specifically (where 3 faces meet).
# These are the corners of face 0-3 at (0,0)/(n-1,0)/(0,n-1)/(n-1,n-1)
# that coincide with corners of face 4/5.  Each cube vertex is shared
# by 3 faces; the discrepancy at a CUBE VERTEX position is what
# matters.  At a cube vertex, the 2-point-avg averages two edge
# halo values each of which comes from a DIFFERENT neighbouring face
# — so the average inherits an O(1) cross-face rotation error.

# Print corner values per face to show the structure.
print("\nPer-face SW cube-corner halo values (padded[:,0,0]) vs interior:")
for f in range(6):
    py_val = B_py[f, 0, 0]
    diag_val = B_py[f, 1, 1]
    print(f"  face{f}: 2-pt avg at (0,0) = {py_val:.4e}  "
          f"interior diagonal (1,1) = {diag_val:.4e}  "
          f"diff = {py_val - diag_val:+.4e}")
