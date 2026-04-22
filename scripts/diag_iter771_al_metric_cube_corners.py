"""Iter-771 diagnostic: measure the A-L gradient metric
coefficients `grad_c00` / `grad_c01` / `grad_c10` / `grad_c11`
at TWO sets of D-grid corners:

1. The 4 CUBE-VERTEX D-grid corners per face (positions (0,0),
   (0,n), (n,0), (n,n) in D-grid indexing shape (n+1, n+1)).
   Iter-771a scope.  NOTE: these D-grid corners update ONLY the
   4 cube-corner CELLS per face (via `_interp_corner_to_center`),
   not the top-4 EDGE peak cells identified by iter-770.

2. The D-grid corners that actually update the top-4 EDGE peak
   cells (e.g. cell (n-2, 0) is updated by the 4 surrounding
   D-grid corners (n-2, 0), (n-2, 1), (n-1, 0), (n-1, 1), none of
   which is a cube-vertex corner).  Iter-771b scope, added per
   Codex stop-time review of iter-771a — "Iter-771 targets the
   wrong D-grid corners for the observed peak cells."

Both sets report `grad_c00..c11` and the ratio (vertex_or_edge /
interior_reference).  Interior reference = median |grad_c*| over
cells strictly `5 <= i,j <= n-5`.

Uses the canonical matrix config.  No dycore integration is
needed — the A-L coefficients are grid-only.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid)


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

c00 = np.asarray(cdgrid.grad_c00)   # (6, n+1, n+1)
c01 = np.asarray(cdgrid.grad_c01)
c10 = np.asarray(cdgrid.grad_c10)
c11 = np.asarray(cdgrid.grad_c11)

# Interior median absolute magnitudes (reference).
interior_mask = np.zeros_like(c00, dtype=bool)
interior_mask[:, 5:-5, 5:-5] = True   # strictly interior
ref_c00 = np.median(np.abs(c00[interior_mask]))
ref_c01 = np.median(np.abs(c01[interior_mask]))
ref_c10 = np.median(np.abs(c10[interior_mask]))
ref_c11 = np.median(np.abs(c11[interior_mask]))

print(f"Interior (strict) median |grad_c*|:")
print(f"  |c00| = {ref_c00:.4e}")
print(f"  |c01| = {ref_c01:.4e}")
print(f"  |c10| = {ref_c10:.4e}")
print(f"  |c11| = {ref_c11:.4e}")
print()

# 4 D-grid corner positions per face (indexing into (n+1, n+1)
# corner array): (0,0), (0,n), (n,0), (n,n).
corners = [("SW", 0, 0), ("NW", 0, n), ("SE", n, 0), ("NE", n, n)]

# Nearby D-grid corners: 1-step in from SW = (1, 0), (0, 1), (1, 1).
def _grad_ratio(c_arr, i, j, ref):
    return c_arr[i, j] / ref if ref > 0 else float("nan")

for face in range(6):
    print(f"Face {face}:")
    for label, ci, cj in corners:
        print(f"  {label}=({ci},{cj}): "
              f"c00={c00[face, ci, cj]:+.4e} "
              f"c01={c01[face, ci, cj]:+.4e} "
              f"c10={c10[face, ci, cj]:+.4e} "
              f"c11={c11[face, ci, cj]:+.4e}  "
              f"|c00|/ref={abs(c00[face,ci,cj])/ref_c00:.2f}x")
    print()

# Summary: report max |c00|/ref ratio across 24 cube-vertex D-grid
# corners (6 faces × 4 corners).
max_ratios = {"c00": 0.0, "c01": 0.0, "c10": 0.0, "c11": 0.0}
for face in range(6):
    for _, ci, cj in corners:
        max_ratios["c00"] = max(max_ratios["c00"],
                                 abs(c00[face, ci, cj]) / ref_c00)
        max_ratios["c01"] = max(max_ratios["c01"],
                                 abs(c01[face, ci, cj]) / max(ref_c01, 1e-30))
        max_ratios["c10"] = max(max_ratios["c10"],
                                 abs(c10[face, ci, cj]) / max(ref_c10, 1e-30))
        max_ratios["c11"] = max(max_ratios["c11"],
                                 abs(c11[face, ci, cj]) / ref_c11)

print("Max |grad_c*|/interior_ref across 24 cube-vertex D-grid corners:")
for k, v in max_ratios.items():
    print(f"  {k}: {v:.2f}x")

# --- Iter-771b: re-measure at the D-grid corners that ACTUALLY
# update the top-4 EDGE peak cells identified by iter-770.
#
# Iter-770 face-local top-4 peak cells (face index and (i, j)):
#   face 0: (34, 0), (34, 35)
#   face 2: (34, 0), (34, 35)
# For each such cell (i, j), the 4 surrounding D-grid corners are
# at indices (i, j), (i, j+1), (i+1, j), (i+1, j+1) in the
# (n+1, n+1) corner array.
print()
print("-- Iter-771b (Codex stop-time): metric coefficients at the")
print("   D-grid corners that update the top-4 EDGE peak cells")
print("   from iter-770 (cells (34, 0) / (34, 35) on faces 0, 2).")
print()
peak_cells = [("face0 (34, 0)",  0, 34, 0),
              ("face0 (34, 35)", 0, 34, 35),
              ("face2 (34, 0)",  2, 34, 0),
              ("face2 (34, 35)", 2, 34, 35)]

edge_max_ratios = {"c00": 0.0, "c01": 0.0, "c10": 0.0, "c11": 0.0}
for label, face, ci, cj in peak_cells:
    print(f"{label} — 4 surrounding D-grid corners:")
    for di in (0, 1):
        for dj in (0, 1):
            corner_i = ci + di
            corner_j = cj + dj
            c00v = c00[face, corner_i, corner_j]
            c01v = c01[face, corner_i, corner_j]
            c10v = c10[face, corner_i, corner_j]
            c11v = c11[face, corner_i, corner_j]
            print(f"  corner ({corner_i:>2},{corner_j:>2}): "
                  f"c00={c00v:+.4e} c01={c01v:+.4e} "
                  f"c10={c10v:+.4e} c11={c11v:+.4e}")
            edge_max_ratios["c00"] = max(edge_max_ratios["c00"],
                                          abs(c00v) / ref_c00)
            edge_max_ratios["c01"] = max(edge_max_ratios["c01"],
                                          abs(c01v) / max(ref_c01, 1e-30))
            edge_max_ratios["c10"] = max(edge_max_ratios["c10"],
                                          abs(c10v) / max(ref_c10, 1e-30))
            edge_max_ratios["c11"] = max(edge_max_ratios["c11"],
                                          abs(c11v) / ref_c11)
    print()

print("Max |grad_c*|/interior_ref across the 16 D-grid corners that")
print("update the iter-770 top-4 EDGE peak cells:")
for k, v in edge_max_ratios.items():
    print(f"  {k}: {v:.2f}x")
