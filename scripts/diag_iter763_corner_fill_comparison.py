"""Iter-763 diagnostic: measure SENSITIVITY of Python's cube-corner
halo-fill output to a simple alternative (interior-diagonal reflect).

**Caveat (iter-763b).**  This diagnostic does NOT compute Fortran's
actual `fill_4corners` formula (sw_core.F90:3856-3915).  That
Fortran fill uses directional copy-from-interior with specific
index mappings (e.g. `q(-1,0) = q(0,2)` for dir=1 SW corner) that
require porting the full Fortran halo convention.  For iter-763's
purpose — establishing whether cube-corner halo fill is a
SENSITIVE part of the solution — this diagnostic uses a simpler
PROXY: "interior-diagonal reflect" (`padded[1, 1]` for the
`padded[0, 0]` corner cell).  If the 2-pt-avg and the proxy differ
materially, the corner fill IS a sensitive knob and iter-764+
porting of Fortran's exact `fill_4corners` is worth the effort.
If they're near-identical, the corner fill is NOT the mode A
driver and another mechanism should be investigated.

Iter-762 showed the residual W2 v-wind artifact is ENTIRELY at
the 8 cube vertices (lat ±35°, lon ±45°/±135°).
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

# Sensitivity check: compare the 2-pt-avg to an "interior-diagonal
# reflect" PROXY (padded[1, 1] for corner padded[0, 0]).  This is
# NOT Fortran's `fill_4corners` formula — which is
#    q(-1, 0) = q(0, 2);  q(0, 0) = q(0, 1)
# (dir=1 at SW corner).  The Fortran formula involves specific
# Fortran-index → Python-padded-index mappings and directional
# sweeps that would require a full port to evaluate here.
#
# The interior-diagonal proxy is merely a CONCRETE alternative
# value — if the 2-pt-avg differs significantly from ANY simple
# alternative, the cube-corner halo fill is a sensitive knob and
# worth porting properly (iter-764+).  A near-identical result
# would indicate the corner fill is NOT the mode A driver.

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
