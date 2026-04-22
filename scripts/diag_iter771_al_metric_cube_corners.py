"""Iter-771 diagnostic: measure the A-L gradient metric
coefficients `grad_c00` / `grad_c01` / `grad_c10` / `grad_c11`
at D-grid corners adjacent to cube vertices vs interior.

Per iter-770's face-local peak localization, mode-A top-4 |v_north|
peaks sit at EDGE cells at Chebyshev distance 1 from cube corners.
These cells are updated by `_interp_corner_to_center(dB_dx, dB_dy)`
from the 4 D-grid corners surrounding each cell.  For an EDGE cell
at (n-2, 0), the 4 surrounding D-grid corners are
  (n-2, 0), (n-2, 1), (n-1, 0), (n-1, 1)  (in D-grid indexing).

The D-grid corner (n-1, 0) is AT the cube vertex.  The A-L
stencil at this corner uses `grad_c00..c11` metric coefficients.

This diagnostic reports:
- `grad_c00/c01/c10/c11` at the D-grid corners at cube vertices
  (i.e. at the 4 cube-corner D-grid positions per face).
- Same coefficients at 3 nearby D-grid corners 1-2 steps in.
- Ratio of cube-vertex values to interior median magnitude.

A large ratio (e.g. >5x) would indicate that the A-L stencil at
the cube-vertex D-grid corner has anomalous metric coefficients
vs the interior.  This would be a candidate structural explanation
for why mode A concentrates near those corners despite iter-765/
766/767/769 corner-fill/smoothing experiments all failing to
reduce the peak.

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
