#!/usr/bin/env python
"""
Diagnostic: compare planar vs spherical area weighting in _spherical_polygon_centroid.

Tests:
  1. Voronoi cells from a coarse icosahedral mesh (subdivision_level=1, 42 cells)
  2. Analytical polar cap (60-90 deg latitude)
  3. Equatorial cell check

Reports angular discrepancy in degrees between:
  - current code  (planar cross-product area)
  - reference     (proper spherical excess area via _spherical_triangle_area)
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from scipy.spatial import SphericalVoronoi

from legoesm.grids.voronoi import (
    _spherical_polygon_centroid,
    _spherical_triangle_area,
    _icosahedral_base,
    _bisect_mesh,
)


# ---------- reference centroid using true spherical areas ---------
def spherical_polygon_centroid_reference(vertices):
    """Centroid using proper spherical triangle areas (spherical excess)."""
    n = len(vertices)
    if n < 3:
        c = vertices.mean(axis=0)
        return c / np.linalg.norm(c)

    v0 = vertices[0]
    total_area = 0.0
    centroid = np.zeros(3)

    for i in range(1, n - 1):
        v1, v2 = vertices[i], vertices[i + 1]
        # Proper spherical triangle area (spherical excess)
        area = _spherical_triangle_area(v0, v1, v2, radius=1.0)
        # Sub-triangle centroid: Cartesian mean projected later
        tri_center = (v0 + v1 + v2) / 3.0
        centroid += area * tri_center
        total_area += area

    if total_area > 0:
        centroid /= total_area
    else:
        centroid = vertices.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-15:
        centroid /= norm
    return centroid


def angular_sep_deg(a, b):
    """Great-circle angle between two unit vectors, in degrees."""
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ====================================================================
#  TEST 1 — Voronoi cells from coarse icosahedral mesh (level 1)
# ====================================================================
print("=" * 72)
print("TEST 1: Voronoi cells from icosahedral subdivision level 1 (42 pts)")
print("=" * 72)

verts, tris = _icosahedral_base()
verts, tris = _bisect_mesh(verts, tris, level=1)

sv = SphericalVoronoi(verts, radius=1.0, center=np.zeros(3))
sv.sort_vertices_of_regions()

errors_deg = []
areas_sr = []  # solid angle of each cell
for i, region in enumerate(sv.regions):
    if len(region) < 3:
        continue
    poly = sv.vertices[region]
    c_current   = _spherical_polygon_centroid(poly)
    c_reference = spherical_polygon_centroid_reference(poly)
    err = angular_sep_deg(c_current, c_reference)
    errors_deg.append(err)

    # Also record cell solid angle for context
    v0 = poly[0]
    cell_area = sum(
        _spherical_triangle_area(v0, poly[j], poly[j + 1])
        for j in range(1, len(poly) - 1)
    )
    areas_sr.append(cell_area)

errors_deg = np.array(errors_deg)
areas_sr = np.array(areas_sr)
cell_angular_size = np.degrees(np.sqrt(areas_sr))  # approx diameter in deg

print(f"  Cells analysed          : {len(errors_deg)}")
print(f"  Cell angular size (deg) : "
      f"min={cell_angular_size.min():.2f}, "
      f"mean={cell_angular_size.mean():.2f}, "
      f"max={cell_angular_size.max():.2f}")
print(f"  Centroid error (deg)    : "
      f"min={errors_deg.min():.6e}, "
      f"mean={errors_deg.mean():.6e}, "
      f"max={errors_deg.max():.6e}")
print(f"  Relative error (err / cell size) : "
      f"min={( errors_deg / cell_angular_size).min():.6e}, "
      f"mean={(errors_deg / cell_angular_size).mean():.6e}, "
      f"max={( errors_deg / cell_angular_size).max():.6e}")


# ====================================================================
#  TEST 2 — Large analytical polar cap (60°-90° N)
# ====================================================================
print()
print("=" * 72)
print("TEST 2: Analytical polar cap (ring 60-90 deg N, 30-deg half-angle)")
print("=" * 72)

# Build a spherical polygon approximating the polar cap boundary at 60 deg N.
n_cap = 64
lat_edge = np.radians(60.0)
lons = np.linspace(0, 2 * np.pi, n_cap, endpoint=False)
cap_verts = np.column_stack([
    np.cos(lat_edge) * np.cos(lons),
    np.cos(lat_edge) * np.sin(lons),
    np.sin(lat_edge) * np.ones(n_cap),
])
# Normalise to unit sphere (should already be, but be safe)
cap_verts /= np.linalg.norm(cap_verts, axis=1, keepdims=True)

# Analytical centroid of a spherical cap from lat_edge to 90 deg
# is at the north pole (0, 0, 1) by symmetry.
analytical_centroid = np.array([0.0, 0.0, 1.0])

c_current   = _spherical_polygon_centroid(cap_verts)
c_reference = spherical_polygon_centroid_reference(cap_verts)

err_current   = angular_sep_deg(c_current,   analytical_centroid)
err_reference = angular_sep_deg(c_reference, analytical_centroid)

# Solid angle of cap: 2 pi (1 - sin(60 deg))
cap_solid_angle = 2.0 * np.pi * (1.0 - np.sin(lat_edge))
cap_diameter_deg = np.degrees(np.sqrt(cap_solid_angle))

print(f"  Cap solid angle         : {cap_solid_angle:.4f} sr "
      f"(~{cap_diameter_deg:.1f} deg diameter)")
print(f"  Current  centroid error : {err_current:.6e} deg "
      f"(from analytic north pole)")
print(f"  Reference centroid error: {err_reference:.6e} deg "
      f"(from analytic north pole)")
print(f"  Difference (current - ref): {err_current - err_reference:.6e} deg")


# ====================================================================
#  TEST 3 — Coarse icosahedral base (12 pts, level=0, very large cells)
# ====================================================================
print()
print("=" * 72)
print("TEST 3: Voronoi cells from icosahedral base (12 pts, level 0)")
print("=" * 72)

verts0, _ = _icosahedral_base()
sv0 = SphericalVoronoi(verts0, radius=1.0, center=np.zeros(3))
sv0.sort_vertices_of_regions()

errors0 = []
areas0 = []
for i, region in enumerate(sv0.regions):
    if len(region) < 3:
        continue
    poly = sv0.vertices[region]
    c_cur = _spherical_polygon_centroid(poly)
    c_ref = spherical_polygon_centroid_reference(poly)
    err = angular_sep_deg(c_cur, c_ref)
    errors0.append(err)
    v0 = poly[0]
    cell_area = sum(
        _spherical_triangle_area(v0, poly[j], poly[j + 1])
        for j in range(1, len(poly) - 1)
    )
    areas0.append(cell_area)

errors0 = np.array(errors0)
areas0 = np.array(areas0)
cell_angular_size0 = np.degrees(np.sqrt(areas0))

print(f"  Cells analysed          : {len(errors0)}")
print(f"  Cell angular size (deg) : "
      f"min={cell_angular_size0.min():.2f}, "
      f"mean={cell_angular_size0.mean():.2f}, "
      f"max={cell_angular_size0.max():.2f}")
print(f"  Centroid error (deg)    : "
      f"min={errors0.min():.6e}, "
      f"mean={errors0.mean():.6e}, "
      f"max={errors0.max():.6e}")
print(f"  Relative error (err / cell size) : "
      f"min={( errors0 / cell_angular_size0).min():.6e}, "
      f"mean={(errors0 / cell_angular_size0).mean():.6e}, "
      f"max={( errors0 / cell_angular_size0).max():.6e}")


# ====================================================================
#  TEST 4 — Scaling: error vs cell size across subdivision levels
# ====================================================================
print()
print("=" * 72)
print("TEST 4: Error scaling across subdivision levels 0-4")
print("=" * 72)
print(f"  {'Level':>5}  {'nCells':>7}  {'Cell size (deg)':>16}  "
      f"{'Max err (deg)':>14}  {'Mean err (deg)':>15}  {'Max rel err':>12}")

for level in range(5):
    v, t = _icosahedral_base()
    if level > 0:
        v, t = _bisect_mesh(v, t, level)
    sv_l = SphericalVoronoi(v, radius=1.0, center=np.zeros(3))
    sv_l.sort_vertices_of_regions()

    errs = []
    cell_sizes = []
    for region in sv_l.regions:
        if len(region) < 3:
            continue
        poly = sv_l.vertices[region]
        c1 = _spherical_polygon_centroid(poly)
        c2 = spherical_polygon_centroid_reference(poly)
        errs.append(angular_sep_deg(c1, c2))
        vv0 = poly[0]
        a = sum(
            _spherical_triangle_area(vv0, poly[j], poly[j + 1])
            for j in range(1, len(poly) - 1)
        )
        cell_sizes.append(np.degrees(np.sqrt(a)))

    errs = np.array(errs)
    cell_sizes = np.array(cell_sizes)
    print(f"  {level:>5}  {len(errs):>7}  {cell_sizes.mean():>16.2f}  "
          f"{errs.max():>14.6e}  {errs.mean():>15.6e}  "
          f"{(errs / cell_sizes).max():>12.6e}")


# ====================================================================
#  SUMMARY
# ====================================================================
print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print("""
The planar cross-product area approximation in _spherical_polygon_centroid
diverges from the proper spherical excess formula. The results above show
how the centroid error scales with cell size.

For typical ESM mesh resolutions (level 3-5, cell sizes ~3-10 deg),
the error is expected to be small relative to the mesh spacing, because
the planar approximation is O(area^2) accurate for small triangles.

For coarse meshes (level 0-1) used in Lloyd relaxation, the error may
be non-trivial since the cells span ~20-60 degrees.
""")
