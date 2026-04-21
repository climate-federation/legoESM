"""Iter-676 metric audit: per-metric boundary vs interior ratio check.

Follow-up to iter-671's audit.  For every remaining cdgrid metric
(rdxa, rdya, dx_edge_y, dy_edge_x, dxa, dya, area_corner, sin_sg,
cos_sg), measure the ratio between cube-boundary values and interior
values.  Any metric with ratio outside [0.5, 2.0] is flagged as a
candidate for iter-666/670-style factor-of-2 bug.

Iter-666 pre-fix: dxc ratio at boundary = 0.5 (half-cell clamp).
Iter-670 pre-fix: area_c ratio at cube vertex = 0.22 (partial quadrant).
Both now match Fortran oracle (ratio ≈ 1).

Iter-676 scope: confirm no additional metrics carry the pattern.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

N = 36
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

def ratio(arr, idx_b, idx_i, name):
    """Ratio of boundary mean to interior mean, with factor-of-2 flag."""
    a = np.asarray(arr)
    b_vals = a[idx_b]
    i_vals = a[idx_i]
    b_mean = float(np.abs(b_vals).mean())
    i_mean = float(np.abs(i_vals).mean())
    r = b_mean / i_mean if i_mean > 0 else float('nan')
    flag = " FLAG" if (r < 0.5 or r > 2.0) else ""
    print(f"  {name:30s}: boundary={b_mean:.4e}  interior={i_mean:.4e}  ratio={r:.4f}{flag}")
    return r

print("=== iter-676 boundary-vs-interior ratio audit ===")
print(f"Grid: C{N}; looking for factor-of-2-style bugs (ratio outside [0.5, 2.0])")
print()

# 1. rdxa, rdya at cell centres (6, n, n)
print("Cell-centre metrics (shape n,n; boundary = rows/cols 0 and n-1):")
f_b = (slice(None), [0, N-1], slice(None))
f_i = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4))
ratio(cdgrid.rdxa, f_b, f_i, "rdxa (i-boundary)")
ratio(cdgrid.rdya, (slice(None), slice(None), [0, N-1]), f_i, "rdya (j-boundary)")

# dxa, dya (the raw ones — inverses of rdxa, rdya)
dxa = 1.0 / np.asarray(cdgrid.rdxa)
dya = 1.0 / np.asarray(cdgrid.rdya)
ratio(dxa, f_b, f_i, "dxa (i-boundary)")
ratio(dya, (slice(None), slice(None), [0, N-1]), f_i, "dya (j-boundary)")

# 2. dx_edge_y (6, n, n+1) at j=0 and j=n
print("\nEdge metrics:")
dxe_i = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4))
ratio(cdgrid.dx_edge_y, (slice(None), slice(None), [0, N]), dxe_i, "dx_edge_y (j=0,n)")

# 3. dy_edge_x (6, n+1, n) at i=0 and i=n
dye_i = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4))
ratio(cdgrid.dy_edge_x, (slice(None), [0, N], slice(None)), dye_i, "dy_edge_x (i=0,n)")

# 4. area_corner (6, n+1, n+1) at all cube boundaries
print("\nCorner area (post iter-670 fix):")
ac = np.asarray(cdgrid.area_corner)
ac_i = (slice(None), slice(N//4, 3*N//4 + 1), slice(N//4, 3*N//4 + 1))
ratio(cdgrid.area_corner, (slice(None), [0, N], slice(1, N)), ac_i, "area_corner (west/east edge)")
ratio(cdgrid.area_corner, (slice(None), slice(1, N), [0, N]), ac_i, "area_corner (south/north edge)")
# Cube vertices (4 corners per face)
verts = (slice(None), [0, N, 0, N], [0, 0, N, N])
ratio(cdgrid.area_corner, verts, ac_i, "area_corner (cube vertices)")

# 5. dxc, dyc (post iter-666 fix)
print("\nCenter-to-center distances (post iter-666 fix):")
dxc = np.asarray(cdgrid.dxc)   # (6, n+1, n)
dyc = np.asarray(cdgrid.dyc)   # (6, n, n+1)
dxc_i = (slice(None), slice(N//4, 3*N//4 + 1), slice(N//4, 3*N//4))
dyc_i = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4 + 1))
ratio(cdgrid.dxc, (slice(None), [0, N], slice(None)), dxc_i, "dxc (i=0,n)")
ratio(cdgrid.dyc, (slice(None), slice(None), [0, N]), dyc_i, "dyc (j=0,n)")

# 6. sin_sg, cos_sg at cube boundaries (6, n, n, 9)
print("\nNon-orthogonality metrics (sin_sg/cos_sg):")
# Pick indices 0,1,2,3 (N,E,W,S edge midpoints); indices 4-7 are corners.
for k, name in [(0, "N"), (1, "E"), (2, "W"), (3, "S")]:
    b = (slice(None), [0, N-1], slice(None), k)
    i = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4), k)
    ratio(cdgrid.sin_sg, b, i, f"sin_sg[:,:,:,{k}] ({name}, i-bdy)")

print()
print("=== end audit ===")
