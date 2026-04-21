"""Iter-764 diagnostic: evaluate the HALO=1-reduced INNER SUBSET of
Fortran `fill_4corners` (sw_core.F90:3856-3915) with best-effort
Python padded-index mapping, and measure the discrepancy from
Python's 2-pt-avg cube-corner fill.

**Scope (iter-764b, important).**  Fortran's fill_4corners writes
TWO halo cells per corner-direction pair:
    dir=1 SW:  q(-1, 0) = q(0, 2)   ← OUTER fill, needs halo≥2
               q( 0, 0) = q(0, 1)   ← INNER fill
Our Python halo=1 padded array has only ONE corner halo cell per
face, so iter-764 evaluates ONLY the inner fill.  A full Fortran
port would need halo≥2 to implement both fills.  The measurement
below is the inner-fill discrepancy only — NOT the full Fortran
formula.

Iter-763b's diagnostic used an "interior-diagonal reflect" PROXY.
Iter-764 replaces the proxy with an evaluation of Fortran's inner
fill rule, which is the best Fortran-formula comparison achievable
at halo=1.

Fortran fill_4corners formula (for halo width matching ours):
  dir=1 (x-sweep):
    SW:  q(-1, 0) = q(0, 2)
         q(0,  0) = q(0, 1)
    SE:  q(npx+1, 0) = q(npx, 2)
         q(npx,   0) = q(npx, 1)
    NW:  q(0,  npy)   = q(0, npy-1)
         q(-1, npy)   = q(0, npy-2)
    NE:  q(npx,   npy) = q(npx, npy-1)
         q(npx+1, npy) = q(npx, npy-2)

  dir=2 (y-sweep):
    SW:  q(0, 0)  = q(1, 0)
         q(0, -1) = q(2, 0)
    SE:  etc.

Python padded array with halo=1 has shape (6, n+2, n+2) indexed
[0..n+1, 0..n+1] with interior at [1..n, 1..n].  Fortran's "q(0, 0)"
is the SW-corner CELL at (i=0, j=0) which would be HALO in our
convention (interior starts at is=1=Python index 1).  Mapping:
  Fortran q(Fi, Fj)  →  our padded[face, Fi+1, Fj+1]
with halo ring at Fortran Fi=0 → Python padded index 1 (one ring in).

Wait — the Fortran halo ring at (-1, 0) maps to Python padded[0, 1]
(i=0, j=1) which is the x-halo strip.  And Fortran q(0, 0) at
(Fi=0, Fj=0) maps to padded[1, 1] which is an INTERIOR cell.
That doesn't match — Fortran's fill_4corners is filling HALO
cells, and q(0, 0) should be a halo cell in their convention.

Re-reading fill_4corners: the array is (isd:ied, jsd:jed) with
isd=1-ng, ied=npx-1+ng.  With ng=3 typical, isd=-2, ied=npx+2.
Then q(0, 0) IS a halo cell (outside is:ie = 1:npx-1).

For our halo=1 case: isd=0, ied=npx, so q(0, 0) IS the outermost
halo cell.  Our Python padded[0, 0] corresponds to Fortran q(isd,
jsd) = q(0, 0) for halo=1.

So the mapping IS:
  Fortran q(Fi, Fj)  →  Python padded[face, Fi, Fj]      (halo=1)

where isd=0, ied=n+1 (Fortran npx=n+1), jsd=0, jed=n+1.

For dir=1 SW:  q(-1, 0) = q(0, 2) would need Fi=-1 which is
OUTSIDE isd=0 — invalid for halo=1.  The formula needs halo≥2.

Hmm — fill_4corners fills TWO halo cells per corner-direction,
which requires halo≥2 to have valid index positions.  With halo=1
our Python only has ONE halo cell per corner.

So for halo=1 we can only fill the OUTER cube-corner cell
padded[face, 0, 0], using the simpler rule:
  dir=1:  padded[0, 0] = padded[0, 1]  (x-edge halo at j=0 inner)
  dir=2:  padded[0, 0] = padded[1, 0]  (y-edge halo at i=0 inner)

Note these DIFFER by direction — exactly Fortran's directional
behavior.  Our 2-pt-avg uses 0.5*(padded[0, 1] + padded[1, 0]),
which is the AVERAGE of the two dir-specific values.
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
B_pad_py = np.asarray(pad_halo(B, interp_offsets=grid.halo_interp_offsets,
                                duogrid=grid.duogrid))
# Shape (6, n+2, n+2).

# Fortran dir=1 fill at SW corner: padded[0, 0] = padded[0, 1]
#   (the x-edge halo cell one step up in j).
# Fortran dir=2 fill at SW corner: padded[0, 0] = padded[1, 0]
#   (the y-edge halo cell one step right in i).
# Similarly for SE/NW/NE corners.

# Extract per-face SW corner info.
print("Per-face cube-corner halo values: Python 2-pt-avg vs Fortran "
      "dir=1 and dir=2 values.\n")
print(f"  {'face':>5}  {'2-pt-avg':>12}  {'dir=1 (y-inner)':>16}  "
      f"{'dir=2 (x-inner)':>16}  {'diff(1-py)':>12}  {'diff(2-py)':>12}")
for f in range(6):
    py_sw = B_pad_py[f, 0, 0]
    dir1_sw = B_pad_py[f, 0, 1]   # x-edge halo at j=1-inner
    dir2_sw = B_pad_py[f, 1, 0]   # y-edge halo at i=1-inner
    print(f"  face{f:>1}  {py_sw:>12.4e}  {dir1_sw:>16.4e}  "
          f"{dir2_sw:>16.4e}  {dir1_sw - py_sw:>+12.3e}  "
          f"{dir2_sw - py_sw:>+12.3e}")

# Global statistics across 24 cube-corner cells (6 faces × 4 corners).
corner_offsets = [(0, 0), (0, -1), (-1, 0), (-1, -1)]  # SW, NW, SE, NE
# Direction-1 (x-sweep) fill: uses the y-edge halo (one step in j toward interior).
# Direction-2 (y-sweep) fill: uses the x-edge halo (one step in i toward interior).
diffs_dir1 = []
diffs_dir2 = []
for f in range(6):
    for ci, cj in corner_offsets:
        # Resolve indices:
        py_val = B_pad_py[f, ci, cj]
        # dir=1: move one step in j toward the interior.
        dj_inner = 1 if cj == 0 else -2
        dir1_val = B_pad_py[f, ci, dj_inner]
        # dir=2: move one step in i toward the interior.
        di_inner = 1 if ci == 0 else -2
        dir2_val = B_pad_py[f, di_inner, cj]
        diffs_dir1.append(dir1_val - py_val)
        diffs_dir2.append(dir2_val - py_val)

diffs_dir1 = np.array(diffs_dir1)
diffs_dir2 = np.array(diffs_dir2)

print(f"\nAcross 24 cube-corner cells:")
print(f"  max |Fortran_dir=1 - Python|: {np.max(np.abs(diffs_dir1)):.4e}")
print(f"  max |Fortran_dir=2 - Python|: {np.max(np.abs(diffs_dir2)):.4e}")
print(f"  max |dir=1 - dir=2| (Fortran self-inconsistency): "
      f"{np.max(np.abs(diffs_dir1 - diffs_dir2)):.4e}")

# By construction, Python's 2-pt-avg = 0.5*(dir1 + dir2), so
# (dir1 - py) = 0.5*(dir1 - dir2) and similarly for dir2.
# Verify:
symmetry_check = np.max(np.abs(diffs_dir1 + diffs_dir2))
print(f"  verify 2-pt-avg symmetry (should be ~0): "
      f"{symmetry_check:.4e}")

B_scale = float(np.max(np.abs(np.asarray(B))))
print(f"\nB field scale: {B_scale:.4e}")
print(f"Relative Fortran-dir1-vs-Python discrepancy: "
      f"{np.max(np.abs(diffs_dir1)) / B_scale:.4%}")
