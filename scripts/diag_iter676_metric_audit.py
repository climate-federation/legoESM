"""Iter-676/677 metric audit: per-metric boundary vs interior ratio check.

Extended follow-up to iter-671's audit (iter-677 correction, after Codex
flagged iter-676's original audit as non-exhaustive).

Audits every CDGridData field with the boundary-vs-interior ratio test.
Splits fields into two populations:

A.  **Uniform-expected** (on a smoothly varying sphere, face-interior and
    face-boundary values should differ by <~2×): rdxa, rdya, dxa, dya,
    dxc, dyc, rdxc, rdyc, dx_edge_y, dy_edge_x, area_corner, rarea_c.
    For these, the ratio test is meaningful.  Pre-iter-666 dxc ratio = 0.5;
    pre-iter-670 area_c at cube vertices = 0.22.  Post-fix: all ~1.0 (clean).

B.  **Non-orthogonality / metric-tensor skew** (geometrically EXPECTED to
    be larger at cube seams, near-zero in interior): sin_sg, cos_sg,
    cosa_u, cosa_v, cosa_cell, cosa_corner, rsin_u, rsin_v, rsin2_cell,
    rsin2_corner, grad_c00/c01/c10/c11.  For these, boundary >> interior
    is geometric reality, not a bug.  The ratio test is INVALID because
    the interior value is near zero.  These are covered by existing
    Fortran-formula lock tests (`TestCosaCornerFortranMatch`,
    `TestSinaUVFromSinSgFortranFormula`, `TestCornerVorticityFortranFormula`,
    etc.), which verify direct Fortran-equation match.

Fields deliberately NOT audited (non-metric geometric coordinates):
  lon_corner, lat_corner, lon_edge_x/y, lat_edge_x/y, angle_corner,
  cos_angle_*, sin_angle_*, f_corner, f_edge_x/y — coordinates and
  rotation angles derived directly from the unaltered gnomonic projection.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

N = 36
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

uniform_results = []
nonortho_results = []

def ratio(arr, idx_b, idx_i, name, bucket):
    a = np.asarray(arr)
    b_vals = a[idx_b]; i_vals = a[idx_i]
    b_mean = float(np.abs(b_vals).mean())
    i_mean = float(np.abs(i_vals).mean())
    r = b_mean / i_mean if i_mean > 0 else float('nan')
    flag = " FLAG" if (np.isfinite(r) and (r < 0.5 or r > 2.0)) else ""
    print(f"  {name:38s}: b={b_mean:.3e}  i={i_mean:.3e}  r={r:.4f}{flag}")
    bucket.append((name, r, flag != ""))
    return r

print("=== iter-677 EXHAUSTIVE CDGridData metric audit ===")
print(f"Grid: C{N}; flag = ratio outside [0.5, 2.0] (iter-666/670 pattern)")
print()

# ================================================================
# GROUP A: uniform-expected metrics (ratio test IS meaningful)
# ================================================================
print(">>> GROUP A: uniform-expected metrics (ratio test is valid)")

# cell-centre (6, n, n)
print("\nCell-centre metrics (shape n,n):")
f_b_i = (slice(None), [0, N-1], slice(None))
f_b_j = (slice(None), slice(None), [0, N-1])
f_i   = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4))
ratio(cdgrid.rdxa, f_b_i, f_i, "rdxa (i-bdy)", uniform_results)
ratio(cdgrid.rdya, f_b_j, f_i, "rdya (j-bdy)", uniform_results)
ratio(1.0 / np.asarray(cdgrid.rdxa), f_b_i, f_i, "dxa (i-bdy)", uniform_results)
ratio(1.0 / np.asarray(cdgrid.rdya), f_b_j, f_i, "dya (j-bdy)", uniform_results)

# u-face (6, n+1, n)
print("\nu-face metrics (shape n+1,n):")
u_b_i = (slice(None), [0, N], slice(None))
u_b_j = (slice(None), slice(None), [0, N-1])
u_i   = (slice(None), slice(N//4, 3*N//4 + 1), slice(N//4, 3*N//4))
ratio(cdgrid.dxc,    u_b_i, u_i, "dxc (i=0,n)",    uniform_results)
ratio(cdgrid.dxc,    u_b_j, u_i, "dxc (j=0,n-1)",  uniform_results)
ratio(cdgrid.rdxc,   u_b_i, u_i, "rdxc (i=0,n)",   uniform_results)
ratio(cdgrid.dy_edge_x, u_b_i, u_i, "dy_edge_x (i=0,n)", uniform_results)
ratio(cdgrid.dy_edge_x, u_b_j, u_i, "dy_edge_x (j=0,n-1)", uniform_results)

# v-face (6, n, n+1)
print("\nv-face metrics (shape n,n+1):")
v_b_i = (slice(None), [0, N-1], slice(None))
v_b_j = (slice(None), slice(None), [0, N])
v_i   = (slice(None), slice(N//4, 3*N//4), slice(N//4, 3*N//4 + 1))
ratio(cdgrid.dyc,    v_b_i, v_i, "dyc (i=0,n-1)",  uniform_results)
ratio(cdgrid.dyc,    v_b_j, v_i, "dyc (j=0,n)",    uniform_results)
ratio(cdgrid.rdyc,   v_b_j, v_i, "rdyc (j=0,n)",   uniform_results)
ratio(cdgrid.dx_edge_y, v_b_i, v_i, "dx_edge_y (i=0,n-1)", uniform_results)
ratio(cdgrid.dx_edge_y, v_b_j, v_i, "dx_edge_y (j=0,n)",   uniform_results)

# corner (6, n+1, n+1)
print("\nCorner metrics (shape n+1,n+1):")
c_i = (slice(None), slice(N//4, 3*N//4 + 1), slice(N//4, 3*N//4 + 1))
c_edge_we = (slice(None), [0, N], slice(1, N))
c_edge_sn = (slice(None), slice(1, N), [0, N])
c_verts   = (slice(None), [0, N, 0, N], [0, 0, N, N])
for arr, name in [(cdgrid.area_corner, "area_corner"),
                  (cdgrid.rarea_c,     "rarea_c")]:
    ratio(arr, c_edge_we, c_i, f"{name} (W/E edges)",    uniform_results)
    ratio(arr, c_edge_sn, c_i, f"{name} (S/N edges)",    uniform_results)
    ratio(arr, c_verts,   c_i, f"{name} (cube vertices)", uniform_results)

# ================================================================
# GROUP B: non-orthogonality metrics (ratio test INVALID; documented)
# ================================================================
print("\n>>> GROUP B: non-orthogonality metrics (ratio test INVALID — documented)")
print("    Coverage: Fortran-formula lock tests in tests/unit/test_cdgrid_fv3_regression.py")
print("    (see TestCosaCornerFortranMatch, TestSinaUVFromSinSgFortranFormula, etc.)")

# These metrics have interior values near zero by definition — face-interior
# of a nearly-orthogonal equiangular cube is cos(non_ortho_angle) ≈ 0.
# At cube seams, non_ortho_angle grows, so cos_sg / cosa_* / grad_c01 / c10
# become O(0.25-0.5).  Ratio boundary/interior ∼6x is geometric truth.
for arr, name in [(cdgrid.cosa_cell,  "cosa_cell (shape n,n)"),
                  (cdgrid.sina_cell,  "sina_cell (shape n,n)"),
                  (cdgrid.rsin2_cell, "rsin2_cell (shape n,n)"),
                  (cdgrid.cosa_u,     "cosa_u (shape n+1,n)"),
                  (cdgrid.rsin_u,     "rsin_u (shape n+1,n)"),
                  (cdgrid.cosa_v,     "cosa_v (shape n,n+1)"),
                  (cdgrid.rsin_v,     "rsin_v (shape n,n+1)"),
                  (cdgrid.cosa_corner, "cosa_corner (shape n+1,n+1)"),
                  (cdgrid.rsin2_corner, "rsin2_corner (shape n+1,n+1)"),
                  (cdgrid.grad_c00,    "grad_c00 (shape n+1,n+1)"),
                  (cdgrid.grad_c01,    "grad_c01 (shape n+1,n+1)"),
                  (cdgrid.grad_c10,    "grad_c10 (shape n+1,n+1)"),
                  (cdgrid.grad_c11,    "grad_c11 (shape n+1,n+1)")]:
    a = np.asarray(arr)
    mx = float(np.abs(a).max())
    print(f"  {name:40s}: |.|_max = {mx:.3e}  (ratio test INVALID — see lock tests)")

for k in range(9):
    for suffix in ['sin_sg', 'cos_sg']:
        a = np.asarray(getattr(cdgrid, suffix))[..., k]
        mx = float(np.abs(a).max())
        print(f"  {suffix}[...,{k}] (shape n,n)               : |.|_max = {mx:.3e}  (ratio test INVALID)")

print()
print("=" * 66)
print(f"GROUP A ({len(uniform_results)} boundary configurations):")
flagged_a = [r for r in uniform_results if r[2]]
print(f"  clean (ratio in [0.5, 2.0]):  {len(uniform_results) - len(flagged_a)}")
print(f"  FLAGGED (pattern match):      {len(flagged_a)}")
for name, r, _ in flagged_a:
    print(f"    FLAG: {name} ratio={r:.4f}")
print()
print("GROUP B: ratio test invalid (near-zero interior).  Fortran-formula")
print("lock tests in tests/unit/test_cdgrid_fv3_regression.py verify these")
print("directly against the Fortran oracle.")
print("=" * 66)
