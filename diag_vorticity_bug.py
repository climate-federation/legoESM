"""Diagnostic script: verify fv3_vorticity at shared cubed-sphere corners.

Tests whether the jnp.pad(..., mode='edge') in fv3_vorticity produces
incorrect vorticity at face-boundary corners by:
1. Creating a C8 cubed-sphere CD-grid
2. Setting up a solid-body rotation wind field (uniform zonal flow)
3. Computing fv3_vorticity
4. Identifying shared physical corners between faces (via Cartesian coords)
5. Comparing vorticity at shared corners
6. Comparing computed vorticity against the analytical value
"""

import sys
sys.path.insert(0, 'src')

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators_cdgrid import fv3_vorticity

# ============================================================
# 1. Create C8 cubed-sphere CD-grid
# ============================================================
N = 8
R = 6.371229e6
OMEGA = 7.292e-5

print("=" * 70)
print(f"FV3 Vorticity Bug Diagnostic — C{N} cubed-sphere")
print("=" * 70)

base = create_cubed_sphere(N, radius=R, omega=OMEGA)
cdgrid = create_cubed_sphere_cdgrid(base, omega=OMEGA)

print(f"Grid created: n={N}, radius={R:.0f} m")
print(f"  dx_edge_y shape: {cdgrid.dx_edge_y.shape}")  # (6, n, n+1)
print(f"  dy_edge_x shape: {cdgrid.dy_edge_x.shape}")  # (6, n+1, n)
print(f"  area_corner shape: {cdgrid.area_corner.shape}")  # (6, n+1, n+1)
print(f"  lon_corner shape: {cdgrid.lon_corner.shape}")
print(f"  lat_corner shape: {cdgrid.lat_corner.shape}")

# ============================================================
# 2. Set up solid-body rotation wind field on D-grid edges
# ============================================================
# For solid-body rotation with U0 = OMEGA * R, the zonal wind is:
#   u_east = U0 * cos(lat),  v_north = 0
# The analytical relative vorticity is:
#   zeta = -2 * U0 * sin(lat) / R = -2 * OMEGA * sin(lat)
# (or equivalently, the absolute vorticity = f + zeta = 0 for retrograde,
#  but we use prograde: zeta = -2*OMEGA*sin(lat) for U0 = OMEGA*R)

U0 = OMEGA * R  # ~464 m/s at equator

# At x-edge midpoints (6, n, n+1): u_d = U0 * cos(lat) * cos(angle)
lat_ex = cdgrid.lat_edge_x  # (6, n, n+1)
angle_ex = cdgrid.angle_edge_x  # (6, n, n+1)
u_d = U0 * jnp.cos(lat_ex) * jnp.cos(angle_ex)

# At y-edge midpoints (6, n+1, n): v_d = -U0 * cos(lat) * sin(angle)
lat_ey = cdgrid.lat_edge_y  # (6, n+1, n)
angle_ey = cdgrid.angle_edge_y  # (6, n+1, n)
v_d = -U0 * jnp.cos(lat_ey) * jnp.sin(angle_ey)

print(f"\nSolid-body rotation: U0 = OMEGA*R = {U0:.2f} m/s")
print(f"  u_d shape: {u_d.shape}, range: [{float(u_d.min()):.2f}, {float(u_d.max()):.2f}]")
print(f"  v_d shape: {v_d.shape}, range: [{float(v_d.min()):.2f}, {float(v_d.max()):.2f}]")

# ============================================================
# 3. Compute fv3_vorticity
# ============================================================
vort = fv3_vorticity(u_d, v_d, cdgrid)
print(f"\nComputed vorticity shape: {vort.shape}")  # (6, n+1, n+1)
print(f"  range: [{float(vort.min()):.6e}, {float(vort.max()):.6e}]")

# ============================================================
# 4. Identify shared physical corners between faces
# ============================================================
# Corner positions in Cartesian (unit sphere) for exact matching
lon_c = np.array(cdgrid.lon_corner)  # (6, n+1, n+1)
lat_c = np.array(cdgrid.lat_corner)

cos_lat_c = np.cos(lat_c)
xc = cos_lat_c * np.cos(lon_c)
yc = cos_lat_c * np.sin(lon_c)
zc = np.sin(lat_c)

# Build flat arrays of (face, i, j, x, y, z)
corners = []
for f in range(6):
    for i in range(N + 1):
        for j in range(N + 1):
            corners.append((f, i, j, xc[f, i, j], yc[f, i, j], zc[f, i, j]))

# Find shared corners: two corners from different faces that are at the
# same physical location (Cartesian distance < threshold)
THRESHOLD = 1e-6  # on unit sphere

shared_pairs = []
n_corners = len(corners)
for a in range(n_corners):
    for b in range(a + 1, n_corners):
        fa, ia, ja, xa, ya, za = corners[a]
        fb, ib, jb, xb, yb, zb = corners[b]
        if fa == fb:
            continue
        dist = np.sqrt((xa - xb)**2 + (ya - yb)**2 + (za - zb)**2)
        if dist < THRESHOLD:
            shared_pairs.append((fa, ia, ja, fb, ib, jb, dist))

print(f"\n{'=' * 70}")
print(f"Shared corners between faces: {len(shared_pairs)} pairs found")
print(f"{'=' * 70}")

# Classify shared corners
boundary_pairs = []  # On face edges (not at cube vertices)
vertex_pairs = []    # At cube vertices (3 faces meet)

for fa, ia, ja, fb, ib, jb, dist in shared_pairs:
    is_vertex_a = (ia in [0, N]) and (ja in [0, N])
    is_vertex_b = (ib in [0, N]) and (jb in [0, N])
    if is_vertex_a and is_vertex_b:
        vertex_pairs.append((fa, ia, ja, fb, ib, jb, dist))
    else:
        boundary_pairs.append((fa, ia, ja, fb, ib, jb, dist))

print(f"  Edge-boundary pairs (2 faces share): {len(boundary_pairs)}")
print(f"  Cube-vertex pairs (3 faces share): {len(vertex_pairs)}")

# ============================================================
# 5. Compare vorticity at shared corners
# ============================================================
vort_np = np.array(vort)

print(f"\n{'=' * 70}")
print("EDGE-BOUNDARY SHARED CORNERS — Vorticity comparison")
print(f"{'=' * 70}")

max_abs_diff = 0.0
max_rel_diff = 0.0
boundary_diffs = []

# Show first 20 pairs and summary
for idx, (fa, ia, ja, fb, ib, jb, dist) in enumerate(boundary_pairs):
    va = vort_np[fa, ia, ja]
    vb = vort_np[fb, ib, jb]
    diff = abs(va - vb)
    avg = 0.5 * (abs(va) + abs(vb))
    rel = diff / avg if avg > 1e-20 else 0.0
    boundary_diffs.append((diff, rel, fa, ia, ja, fb, ib, jb, va, vb))
    if diff > max_abs_diff:
        max_abs_diff = diff
    if rel > max_rel_diff:
        max_rel_diff = rel

# Sort by absolute difference (largest first)
boundary_diffs.sort(key=lambda x: -x[0])

print(f"\nTop 20 worst discrepancies (sorted by |diff|):")
print(f"{'Face A':>8s} {'(i,j)':>8s} {'Face B':>8s} {'(i,j)':>8s}"
      f"  {'vort_A':>12s} {'vort_B':>12s} {'|diff|':>12s} {'rel_diff':>10s}")
print("-" * 90)
for diff, rel, fa, ia, ja, fb, ib, jb, va, vb in boundary_diffs[:20]:
    print(f"  {fa:>5d}  ({ia:>2d},{ja:>2d})  {fb:>5d}  ({ib:>2d},{jb:>2d})"
          f"  {va:>12.6e} {vb:>12.6e} {diff:>12.6e} {rel:>10.4f}")

print(f"\nSummary for edge-boundary pairs:")
print(f"  Max absolute difference: {max_abs_diff:.6e}")
print(f"  Max relative difference: {max_rel_diff:.4f} ({max_rel_diff*100:.2f}%)")
diffs_arr = np.array([d[0] for d in boundary_diffs])
print(f"  Mean absolute difference: {np.mean(diffs_arr):.6e}")
print(f"  Median absolute difference: {np.median(diffs_arr):.6e}")

# Count how many are exact matches vs discrepant
n_exact = sum(1 for d in boundary_diffs if d[0] < 1e-20)
n_close = sum(1 for d in boundary_diffs if d[0] < 1e-10)
n_bad = sum(1 for d in boundary_diffs if d[0] > 1e-6)
print(f"  Exact matches (|diff| < 1e-20): {n_exact}/{len(boundary_diffs)}")
print(f"  Close matches (|diff| < 1e-10): {n_close}/{len(boundary_diffs)}")
print(f"  Significant discrepancies (|diff| > 1e-6): {n_bad}/{len(boundary_diffs)}")

# ============================================================
# 6. Compare computed vs analytical vorticity
# ============================================================
# Analytical: zeta = -2 * OMEGA * sin(lat) at corner positions
lat_corner = np.array(cdgrid.lat_corner)
zeta_analytical = -2.0 * OMEGA * np.sin(lat_corner)

print(f"\n{'=' * 70}")
print("ANALYTICAL COMPARISON: computed vs -2*OMEGA*sin(lat)")
print(f"{'=' * 70}")

err = np.abs(vort_np - zeta_analytical)
print(f"\nGlobal error statistics:")
print(f"  Max absolute error:  {np.max(err):.6e} s^-1")
print(f"  Mean absolute error: {np.mean(err):.6e} s^-1")
print(f"  RMS error:          {np.sqrt(np.mean(err**2)):.6e} s^-1")
print(f"  Analytical range: [{np.min(zeta_analytical):.6e}, {np.max(zeta_analytical):.6e}]")

# Relative error (avoid division by zero near equator)
mask = np.abs(zeta_analytical) > 1e-10
rel_err = np.zeros_like(err)
rel_err[mask] = err[mask] / np.abs(zeta_analytical[mask])
print(f"  Max relative error (|zeta_ana| > 1e-10): {np.max(rel_err[mask]):.4f}"
      f" ({np.max(rel_err[mask])*100:.2f}%)")
print(f"  Mean relative error: {np.mean(rel_err[mask]):.4f}"
      f" ({np.mean(rel_err[mask])*100:.2f}%)")

# Separate interior vs boundary corners
print(f"\nInterior corners (1..n-1, 1..n-1) vs boundary corners:")
interior_mask = np.zeros((6, N+1, N+1), dtype=bool)
interior_mask[:, 1:N, 1:N] = True
boundary_mask = ~interior_mask

err_int = err[interior_mask]
err_bnd = err[boundary_mask]
print(f"  Interior: max={np.max(err_int):.6e}, mean={np.mean(err_int):.6e}"
      f" ({np.sum(interior_mask)} corners)")
print(f"  Boundary: max={np.max(err_bnd):.6e}, mean={np.mean(err_bnd):.6e}"
      f" ({np.sum(boundary_mask)} corners)")

# Relative error, interior vs boundary
if np.any(interior_mask & mask):
    rel_int = rel_err[interior_mask & mask]
    print(f"  Interior relative: max={np.max(rel_int):.4f}, mean={np.mean(rel_int):.4f}")
if np.any(boundary_mask & mask):
    rel_bnd = rel_err[boundary_mask & mask]
    print(f"  Boundary relative: max={np.max(rel_bnd):.4f}, mean={np.mean(rel_bnd):.4f}")

# Now look specifically at shared corners: which face gives the better answer?
print(f"\n{'=' * 70}")
print("SHARED CORNER ANALYSIS: which face gives closer-to-analytical?")
print(f"{'=' * 70}")

n_a_better = 0
n_b_better = 0
n_tied = 0
for diff, rel, fa, ia, ja, fb, ib, jb, va, vb in boundary_diffs:
    ana = zeta_analytical[fa, ia, ja]  # same physical point
    err_a = abs(va - ana)
    err_b = abs(vb - ana)
    if abs(err_a - err_b) < 1e-20:
        n_tied += 1
    elif err_a < err_b:
        n_a_better += 1
    else:
        n_b_better += 1

print(f"  Face A closer to analytical: {n_a_better}")
print(f"  Face B closer to analytical: {n_b_better}")
print(f"  Tied: {n_tied}")

# Show worst shared-corner analytical errors
print(f"\nWorst shared-corner pairs by max error vs analytical:")
print(f"{'Face A':>8s} {'(i,j)':>8s} {'err_A':>12s}  "
      f"{'Face B':>8s} {'(i,j)':>8s} {'err_B':>12s}  {'analytical':>12s}")
print("-" * 90)
worst_ana = []
for diff, rel, fa, ia, ja, fb, ib, jb, va, vb in boundary_diffs[:20]:
    ana = zeta_analytical[fa, ia, ja]
    err_a = abs(va - ana)
    err_b = abs(vb - ana)
    worst_ana.append((max(err_a, err_b), fa, ia, ja, err_a, fb, ib, jb, err_b, ana))
worst_ana.sort(key=lambda x: -x[0])
for _, fa, ia, ja, err_a, fb, ib, jb, err_b, ana in worst_ana[:20]:
    print(f"  {fa:>5d}  ({ia:>2d},{ja:>2d}) {err_a:>12.6e}  "
          f"  {fb:>5d}  ({ib:>2d},{jb:>2d}) {err_b:>12.6e}  {ana:>12.6e}")

# ============================================================
# 7. Verdict
# ============================================================
print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")

BUG_THRESHOLD_ABS = 1e-8  # vorticity mismatch at shared corners
BUG_THRESHOLD_REL = 0.01  # 1% relative difference

if max_abs_diff > BUG_THRESHOLD_ABS or max_rel_diff > BUG_THRESHOLD_REL:
    print(f"\n  ** BUG CONFIRMED **")
    print(f"  fv3_vorticity produces DIFFERENT values at physically identical")
    print(f"  corners on different faces.")
    print(f"  Max absolute mismatch: {max_abs_diff:.6e} s^-1")
    print(f"  Max relative mismatch: {max_rel_diff*100:.2f}%")
    print(f"  This is caused by jnp.pad(mode='edge') repeating same-face")
    print(f"  edge values instead of using neighboring-face edge data.")
else:
    print(f"\n  Bug NOT confirmed at C{N} resolution.")
    print(f"  Max absolute mismatch: {max_abs_diff:.6e} s^-1")
    print(f"  Max relative mismatch: {max_rel_diff*100:.4f}%")
    print(f"  Shared corners agree to within tolerance.")

# Also check if boundary analytical error is significantly worse
ratio_bnd_int = np.mean(err_bnd) / max(np.mean(err_int), 1e-20)
print(f"\n  Boundary/interior mean error ratio: {ratio_bnd_int:.2f}x")
if ratio_bnd_int > 2.0:
    print(f"  Boundary errors are {ratio_bnd_int:.1f}x worse than interior,")
    print(f"  consistent with incorrect boundary treatment in fv3_vorticity.")
