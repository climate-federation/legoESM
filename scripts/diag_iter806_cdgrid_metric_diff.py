"""Iter-806 diagnostic: compare cdgrid metric fields between
LEGACY and DUOGRID to find the DUOGRID dh/dt blowup source.

Iter-803-805 ruled out all halo paths (fill_corner_region,
cube_rmp_vectorized, interp_offsets, pad_halo) as the dh/dt
blowup cause.  Remaining candidate: cdgrid metric fields
themselves differ between LEGACY and DUOGRID because cdgrid
creation uses `base.duogrid` to remap coordinate fields.

Iter-806 compares every metric field exposed by cdgrid between
the two paths, looking for cube-vertex-local differences.

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
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist(lat_r, lon_r):
    min_d = np.inf
    for vlat in CUBE_VERTEX_LATS_R:
        for vlon in CUBE_VERTEX_LONS_R:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return min_d


def _build_cdgrid(use_duogrid, n):
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    model = FV3EdgeShallowWaterModel(grid)
    return model.cdgrid, grid


n = 36
L_cdg, L_grid = _build_cdgrid(use_duogrid=False, n=n)
D_cdg, D_grid = _build_cdgrid(use_duogrid=True, n=n)

# List of CDGrid metric attribute names to compare.
# From CubedSphereCDGrid NamedTuple fields.
candidate_attrs = [
    'cosa_u', 'cosa_v', 'cosa_s',
    'rsin_u', 'rsin_v', 'rsin2_cell',
    'rsin2_corner', 'cosa_corner',
    'dx_edge_y', 'dy_edge_x',
    'dxc', 'dyc', 'area_corner',
    'lat_edge_x', 'lat_edge_y',
    'lon_edge_x', 'lon_edge_y',
    'cos_angle_edge_x', 'sin_angle_edge_x',
    'cos_angle_edge_y', 'sin_angle_edge_y',
    'cos_sg', 'sin_sg',
]

base_attrs = [
    'area', 'lat', 'lon', 'sin_lat', 'cos_lat',
    'angle', 'cos_angle', 'sin_angle',
    'dx', 'dy',
    'cos_angle_padded', 'sin_angle_padded',
]

print(f"Iter-806 cdgrid metric field comparison LEGACY vs DUOGRID at C{n}")
print(f"(Looking for fields where DUOGRID differs from LEGACY)")
print()
print(f"{'field':>22}  {'max |ΔL-D|':>12}  {'max|L|':>12}  {'rel':>10}  {'status':>8}")
print("-" * 75)

differences = []
for attr_name in candidate_attrs:
    try:
        L = np.asarray(getattr(L_cdg, attr_name))
        D = np.asarray(getattr(D_cdg, attr_name))
    except AttributeError:
        continue
    if L.shape != D.shape:
        print(f"  {attr_name:>20}  SHAPE MISMATCH  L:{L.shape}  D:{D.shape}")
        continue
    dmax = float(np.max(np.abs(L - D)))
    lmax = float(np.max(np.abs(L))) if np.max(np.abs(L)) > 0 else 1.0
    rel = dmax / max(lmax, 1e-30)
    status = "MATCH" if dmax < 1e-12 else ("close" if rel < 1e-6 else "DIFFER")
    print(f"  cdgrid.{attr_name:>14}  {dmax:>12.3e}  {lmax:>12.3e}  "
          f"{rel:>10.3e}  {status:>8}")
    if dmax > 1e-12:
        differences.append((attr_name, dmax, rel))

for attr_name in base_attrs:
    try:
        L = np.asarray(getattr(L_grid, attr_name))
        D = np.asarray(getattr(D_grid, attr_name))
    except AttributeError:
        continue
    if L.shape != D.shape:
        print(f"  base.{attr_name:>20}  SHAPE MISMATCH  L:{L.shape}  D:{D.shape}")
        continue
    dmax = float(np.max(np.abs(L - D)))
    lmax = float(np.max(np.abs(L))) if np.max(np.abs(L)) > 0 else 1.0
    rel = dmax / max(lmax, 1e-30)
    status = "MATCH" if dmax < 1e-12 else ("close" if rel < 1e-6 else "DIFFER")
    print(f"  base.{attr_name:>16}  {dmax:>12.3e}  {lmax:>12.3e}  "
          f"{rel:>10.3e}  {status:>8}")
    if dmax > 1e-12:
        differences.append((attr_name, dmax, rel))

print()
if differences:
    print(f"Fields that DIFFER between LEGACY and DUOGRID ({len(differences)}):")
    for attr_name, dmax, rel in sorted(differences, key=lambda x: -x[1]):
        print(f"  {attr_name}: max |Δ|={dmax:.3e}, rel={rel:.3e}")
else:
    print(f"All checked fields MATCH between LEGACY and DUOGRID.")

print()
print("Interpretation cues (observational only):")
print("- If any metric field differs: that's a candidate for the")
print("  DUOGRID dh/dt blowup source (since halos are ruled out).")
print("- If ALL fields match: the dh/dt difference is purely in")
print("  cgrid_mass_flux_divergence's internal logic, not the")
print("  cdgrid it reads from.")
