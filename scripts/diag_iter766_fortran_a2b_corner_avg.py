"""Iter-766 diagnostic: measure the Fortran `a2b_ord4` 3-point
cube-corner average (a2b_edge.F90:385-388) vs Python's current
2-point edge-halo-only average (`_fill_corners_h1` in `halo.py`).

**Fortran reference.**  `a2b_ord4` (model/a2b_edge.F90) interpolates
A-grid scalars to B-grid vertices for the pressure-gradient machinery.
The interior (bulk) stencil is a 4-point average (line 380):

    qout(i,j) = 0.25*(qin(i-1,j-1) + qin(i,j-1)
                      + qin(i-1,j) + qin(i,j))

At the 4 CUBE VERTICES (lines 385-388), Fortran OVERRIDES this with
a 3-point formula that EXCLUDES the cube-corner A-halo cell:

    if ( sw_corner ) qout(1,1) = r3*(qin(1,1) + qin(1,0) + qin(0,1))

where `r3 = 1.0/3.0`.  The 3 included cells are:
    qin(1, 1)  — first interior A-cell adjacent to the cube vertex
    qin(1, 0)  — south halo strip at i=1
    qin(0, 1)  — west halo strip at j=1

The EXCLUDED cell is qin(0, 0) — the cube-corner halo cell (the
diagonal A-cell that would have been the 4th term of the 4-point
bulk formula).

**Our Python A-L gradient path.**  `_arakawa_lamb_gradient` uses a
4-point stencil at ALL D-grid corners INCLUDING the cube vertex.
At the cube-vertex D-grid corner, the 4 reads are:
    padded[:, 0, 0] — cube-corner halo (filled by 2-pt-avg)
    padded[:, 0, 1] — west halo at j=1
    padded[:, 1, 0] — south halo at i=1
    padded[:, 1, 1] — first interior cell

Python's 2-pt-avg fills padded[:, 0, 0] = 0.5*(padded[:, 0, 1]
+ padded[:, 1, 0]) — edge halos only, midpoint of the two
direction-specific Fortran inner fills (iter-764 confirmed).

**Iter-766 candidate.**  Replace the 2-pt-avg with Fortran's
a2b_ord4 3-point corner average:

    padded[:, 0, 0] = (1/3)*(padded[:, 0, 1]
                             + padded[:, 1, 0]
                             + padded[:, 1, 1])

This includes the diagonal interior cell (padded[:, 1, 1]) that
Fortran uses, giving lower weight (1/3 vs 1/2) to the edge halos.

Iter-766's question: does replacing 2-pt-avg with Fortran's
a2b_ord4 3-pt-avg reduce the W2 v-wind mode A artifact at cube
vertices?  If yes, wire it up behind a `fortran_a2b_corner_avg`
flag, test, and set as default.  If no, document as falsified.

This diagnostic measures the magnitude of the difference on the
W2-exact KE field (no runtime change; plan for iter-766 code
change if the magnitude is substantial).
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo


n = 36
grid = create_cubed_sphere(n)

# W2-exact-IC smooth KE field.
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
lat = np.asarray(grid.lat, dtype=np.float64)
u_east = u0 * np.cos(lat)
KE = 0.5 * u_east ** 2
B = jnp.asarray(KE)

# Python 2-pt-avg padded (current behaviour).
B_pad = np.asarray(pad_halo(B, interp_offsets=grid.halo_interp_offsets,
                             duogrid=grid.duogrid))
# Shape (6, n+2, n+2).

print("Per-face cube-corner halo values: Python 2-pt-avg "
      "(current) vs Fortran a2b_ord4 3-pt-avg (proposed).\n"
      "Fortran a2b_ord4 at SW cube vertex (a2b_edge.F90:385):\n"
      "  qout(1,1) = r3*(qin(1,1) + qin(1,0) + qin(0,1))\n"
      "i.e. 3-pt average EXCLUDING qin(0,0) — the diagonal halo.\n\n"
      "Python 2-pt-avg: padded[0,0] = 0.5*(padded[0,1]+padded[1,0])\n"
      "Fortran 3-pt:    padded[0,0] = (1/3)*(padded[0,1]+padded[1,0]+padded[1,1])\n"
      "(Difference: 3-pt adds the interior diagonal cell with 1/3 "
      "weight and lowers edge halo weights from 1/2 to 1/3.)\n")

print(f"  {'face':>5}  {'2-pt-avg':>12}  {'3-pt-avg':>12}  "
      f"{'diff':>12}  {'rel%':>8}")
for f in range(6):
    py_sw = B_pad[f, 0, 0]
    fortran_sw = (1.0/3.0) * (B_pad[f, 0, 1] + B_pad[f, 1, 0]
                               + B_pad[f, 1, 1])
    diff = fortran_sw - py_sw
    rel = 100.0 * diff / max(abs(py_sw), 1e-12)
    print(f"  face{f:>1}  {py_sw:>12.4e}  {fortran_sw:>12.4e}  "
          f"{diff:>+12.3e}  {rel:>+7.3f}%")

# Global statistics across 24 cube-corner cells (6 faces × 4 corners).
def _fortran_a2b_3pt(B_pad, corner):
    """Fortran a2b_ord4 3-pt-avg for a given cube corner (SW/SE/NW/NE)."""
    if corner == "SW":
        # padded[:, 0, 0] = (1/3)*(padded[:, 0, 1] + padded[:, 1, 0] + padded[:, 1, 1])
        return (1.0/3.0) * (B_pad[:, 0, 1] + B_pad[:, 1, 0]
                             + B_pad[:, 1, 1])
    elif corner == "SE":
        # padded[:, -1, 0] = (1/3)*(padded[:, -2, 0] + padded[:, -1, 1] + padded[:, -2, 1])
        return (1.0/3.0) * (B_pad[:, -2, 0] + B_pad[:, -1, 1]
                             + B_pad[:, -2, 1])
    elif corner == "NW":
        # padded[:, 0, -1] = (1/3)*(padded[:, 0, -2] + padded[:, 1, -1] + padded[:, 1, -2])
        return (1.0/3.0) * (B_pad[:, 0, -2] + B_pad[:, 1, -1]
                             + B_pad[:, 1, -2])
    elif corner == "NE":
        # padded[:, -1, -1] = (1/3)*(padded[:, -2, -1] + padded[:, -1, -2] + padded[:, -2, -2])
        return (1.0/3.0) * (B_pad[:, -2, -1] + B_pad[:, -1, -2]
                             + B_pad[:, -2, -2])

def _current_2pt(B_pad, corner):
    """Current Python 2-pt edge-halo-only average."""
    if corner == "SW":
        return 0.5 * (B_pad[:, 0, 1] + B_pad[:, 1, 0])
    elif corner == "SE":
        return 0.5 * (B_pad[:, -1, 1] + B_pad[:, -2, 0])
    elif corner == "NW":
        return 0.5 * (B_pad[:, 0, -2] + B_pad[:, 1, -1])
    elif corner == "NE":
        return 0.5 * (B_pad[:, -1, -2] + B_pad[:, -2, -1])

diffs = []
for corner_name in ("SW", "SE", "NW", "NE"):
    fortran_vals = _fortran_a2b_3pt(B_pad, corner_name)
    current_vals = _current_2pt(B_pad, corner_name)
    for f in range(6):
        diffs.append(fortran_vals[f] - current_vals[f])

diffs = np.array(diffs)

print(f"\nAcross 24 cube-corner cells (6 faces × 4 corners):")
print(f"  max |Fortran 3-pt-avg − Python 2-pt-avg|: "
      f"{np.max(np.abs(diffs)):.4e}")
print(f"  rms (Fortran 3-pt-avg − Python 2-pt-avg): "
      f"{np.sqrt(np.mean(diffs**2)):.4e}")

B_scale = float(np.max(np.abs(np.asarray(B))))
print(f"\nB field scale: {B_scale:.4e}")
print(f"Relative max discrepancy: "
      f"{np.max(np.abs(diffs)) / B_scale:.4%}")

# Detailed per-corner-per-face report for the SW family.
print("\nPer-face SW cube-corner values (detailed):")
print(f"  {'face':>5}  {'pad[0,1]':>12}  {'pad[1,0]':>12}  "
      f"{'pad[1,1]':>12}  {'py 2-pt':>12}  {'F 3-pt':>12}  {'diff':>12}")
for f in range(6):
    p01 = B_pad[f, 0, 1]
    p10 = B_pad[f, 1, 0]
    p11 = B_pad[f, 1, 1]
    py = 0.5 * (p01 + p10)
    fortran = (1.0/3.0) * (p01 + p10 + p11)
    print(f"  face{f:>1}  {p01:>12.4e}  {p10:>12.4e}  "
          f"{p11:>12.4e}  {py:>12.4e}  {fortran:>12.4e}  "
          f"{fortran - py:>+12.3e}")
