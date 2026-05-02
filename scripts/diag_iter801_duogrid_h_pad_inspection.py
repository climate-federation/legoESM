"""Iter-801 diagnostic: inspect h_pad cube-corner values under
DUOGRID vs LEGACY at halo=2.

Per iter-800's finding that the DUOGRID blowup is in pad_halo(h,
halo=2, duogrid=dg), iter-801 inspects the padded h-field cells
at cube-vertex halo positions and compares to LEGACY.

For halo=2 with W2 solid-body rotation, h is smooth across cube
vertices with characteristic values ~2900 m (mean) ± a few m
deviation.  The 4 cube-vertex halo cells per face (at padded
indices [0:2, 0:2], [-2:, 0:2], [0:2, -2:], [-2:, -2:]) should
contain physically-sensible values close to the analytical h.

If DUOGRID produces nonsensical values (e.g., 1e6 or 1e-30) at
cube-vertex halo cells, `cube_rmp_vectorized` or
`fill_corner_region` is broken.

Scope: observational only.
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
from legoesm.grids.halo import pad_halo
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _run(use_duogrid, n):
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets_h2
    sw = williamson_test2(grid)
    h = sw.h.data
    h_pad = pad_halo(h, halo=2, interp_offsets=offsets, duogrid=dg)
    return {
        'h_pad': np.asarray(h_pad),
        'grid': grid,
        'h_interior': np.asarray(h),
    }


n = 36
r_legacy = _run(use_duogrid=False, n=n)
r_duogrid = _run(use_duogrid=True, n=n)

h_pad_L = r_legacy['h_pad']
h_pad_D = r_duogrid['h_pad']

print(f"Iter-801 h_pad cube-corner inspection at C{n}, halo=2, W2 IC")
print()
print(f"Interior h: min={r_legacy['h_interior'].min():.2f}, "
      f"max={r_legacy['h_interior'].max():.2f}, "
      f"mean={r_legacy['h_interior'].mean():.2f}")
print(f"h_pad shape: {h_pad_L.shape} (halo=2, so (6, n+4, n+4) = "
      f"(6, {n+4}, {n+4}))")
print()

# The 4 cube-vertex halo cells per face are at padded indices:
#   SW corner: [0:2, 0:2] (2x2 block)
#   SE corner: [-2:, 0:2]
#   NW corner: [0:2, -2:]
#   NE corner: [-2:, -2:]

corners = [
    ('SW', slice(0, 2), slice(0, 2)),
    ('SE', slice(-2, None), slice(0, 2)),
    ('NW', slice(0, 2), slice(-2, None)),
    ('NE', slice(-2, None), slice(-2, None)),
]

print(f"{'face':>4}  {'corner':>6}  {'cells':>10}  "
      f"{'LEGACY min':>12}  {'LEGACY max':>12}  "
      f"{'DUOGRID min':>12}  {'DUOGRID max':>12}  {'max |Δ|':>10}")
print("-" * 110)
for face in range(6):
    for label, si, sj in corners:
        L_block = h_pad_L[face, si, sj]
        D_block = h_pad_D[face, si, sj]
        diff = np.abs(L_block - D_block)
        max_diff = float(np.max(diff))
        print(f"  {face:>2}  {label:>6}  {'2x2':>10}  "
              f"{float(L_block.min()):>12.2f}  {float(L_block.max()):>12.2f}  "
              f"{float(D_block.min()):>12.2f}  {float(D_block.max()):>12.2f}  "
              f"{max_diff:>10.3e}")

print()
print("Summary:")
max_diff_all = float(np.max(np.abs(h_pad_L - h_pad_D)))
idx = np.unravel_index(np.argmax(np.abs(h_pad_L - h_pad_D)), h_pad_L.shape)
face, i, j = int(idx[0]), int(idx[1]), int(idx[2])
print(f"  max |h_pad_L - h_pad_D| over entire padded array = {max_diff_all:.3e}")
print(f"  at face={face}, padded (i,j)=({i},{j})")
print(f"  LEGACY[face,{i},{j}] = {h_pad_L[face, i, j]:.4f}")
print(f"  DUOGRID[face,{i},{j}] = {h_pad_D[face, i, j]:.4f}")
print(f"  interior h near this position: min {float(r_legacy['h_interior'][face].min()):.2f}, max {float(r_legacy['h_interior'][face].max()):.2f}")

print()
print("Interpretation cues:")
print("- If max |Δ| is small (< 10) at all cube corners: halo")
print("  values are similar between paths; the DUOGRID PPM")
print("  transport blowup must come from PPM stencil reading")
print("  edge-halo cells (not cube-corner halo).")
print("- If max |Δ| is LARGE (> 100) at some cube corner: the")
print("  DUOGRID halo is grossly wrong at that corner.")
