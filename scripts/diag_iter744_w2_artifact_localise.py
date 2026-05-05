"""Iter-744 diagnostic: localise the W2 v-wind artifact peaks in
(lon, lat) to narrow the production bug target.

Iter-717 user-reported artifact is "mode-4 polar" with ±0.3 m/s
bands.  Iter-742 confirmed v_ll_Linf = 0.303 m/s on the regridded
v_ll field that the PNG plots.  Iter-744 reports the SPATIAL
LOCATION of the peak to differentiate between:

  (a) 8 cube corners (known weak points on a cubed sphere; each
      cube vertex sits at a specific (lon, lat) on the sphere).
  (b) Panel-edge midpoints (4 per face x 6 = 24 on the sphere).
  (c) Polar caps (2 centres at lat ±90).
  (d) Other structure (mid-panel interior, etc.).

Cube-corner locations on a standard FV3 cubed sphere:
  Face 0-3 (equatorial) NW corner lats ~ ±35.3° (arctan(1/sqrt(2))).
  Face 4 (N pole) sits at lat +90 with 4 corners at lat ~+35.3°.
  Face 5 (S pole) sits at lat -90 with 4 corners at lat ~-35.3°.

If the peak is at those ±35.3° latitudes, it's cube-corner
localized.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter744_w2_artifact_localise.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig)
from tests.test_cases.williamson import williamson_test2
from scripts.run_atmosphere_test_matrix import (
    _hyperdiff_cube, _div_damp_cube, _regrid_2d)


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

dt = 300.0
config = CDGridShallowWaterConfig(
    hyperdiff_coeff=_hyperdiff_cube(n),
    div_damp=_div_damp_cube(n),
    boundary_fix=True)
model = FV3EdgeShallowWaterModel(grid, config)
cdgrid = model.cdgrid

sw = williamson_test2(grid)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(
    h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
model.set_initial_mass(state)

print("Running W2 C36 1 day...")
for _ in range(int(86400 / dt)):
    state = model.step(state, dt)

ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
ca_np = np.asarray(ca_4edge, dtype=np.float64)
sa_np = np.asarray(sa_4edge, dtype=np.float64)
u_cc = 0.5 * (np.asarray(state.u_d, dtype=np.float64)[:, :, :-1]
              + np.asarray(state.u_d, dtype=np.float64)[:, :, 1:])
v_cc = 0.5 * (np.asarray(state.v_d, dtype=np.float64)[:, :-1, :]
              + np.asarray(state.v_d, dtype=np.float64)[:, 1:, :])
v_north_face = sa_np * u_cc + ca_np * v_cc

lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, "cube")

print(f"\nv_ll shape: {v_ll.shape}")
print(f"max|v_ll| = {np.max(np.abs(v_ll)):.4e}")

# --- Top-N peaks in (lat, lon) ---
# v_ll is (nlat, nlon) in the regrid convention — check via shape.
# _regrid_2d outputs an interpolated field on a 2D lat-lon mesh.
# Default resolution is set internally; check by looking at the
# regrid function signature and comparing to lon_deg/lat_deg.
abs_v = np.abs(v_ll)
flat = abs_v.flatten()
top_k = 12  # 8 cube corners + 2 poles + few edge-midpoints
idx = np.argsort(flat)[-top_k:][::-1]

# Figure out the regrid mesh.  _regrid_2d("cube") outputs on a
# target lat-lon mesh; peek at its likely resolution by comparing
# v_ll.shape against lon_deg/lat_deg.  The regridder typically
# uses a target mesh like (181, 360) or (91, 180).
lat_target = np.linspace(-90, 90, v_ll.shape[0])
lon_target = np.linspace(-180, 180, v_ll.shape[1], endpoint=False)

print(f"\nTop {top_k} peaks of |v_ll| (regridded to "
      f"{v_ll.shape} lat-lon mesh):")
print(f"  {'rank':>4}  {'lat':>7}  {'lon':>8}  {'v_ll':>10}  {'sign':>5}")
for k, ix in enumerate(idx):
    i, j = np.unravel_index(ix, v_ll.shape)
    print(f"  {k+1:>4}  {lat_target[i]:>7.2f}  {lon_target[j]:>8.2f}  "
          f"{flat[ix]:>10.4e}  {np.sign(v_ll[i, j]):>5.0f}")

# --- Cube-corner location check ---
print("\nExpected cube-corner latitudes: ±35.3° (arctan(1/sqrt(2))).")
print("If top peaks cluster near ±35.3° in lat, the artifact is cube-corner-localized.")
print("If they cluster near ±60° (panel-edge midpoints) or ±90° (poles), it's different structure.")

# --- Latitude histogram of peaks ---
top_lats = np.array([lat_target[np.unravel_index(ix, v_ll.shape)[0]]
                     for ix in idx])
print(f"\nPeak latitude distribution: "
      f"mean={top_lats.mean():.1f}°, "
      f"range=[{top_lats.min():.1f}°, {top_lats.max():.1f}°]")
cube_corner_lat = np.degrees(np.arctan(1.0 / np.sqrt(2.0)))
print(f"Mean |peak lat| = {np.mean(np.abs(top_lats)):.2f}° vs "
      f"cube-corner ±{cube_corner_lat:.2f}°")

# --- Zonal symmetry check ---
# If the 4 cube corners per hemisphere project to 4 specific longitudes
# for faces 0/1/2/3 (eastern edges at lons 0, 90, 180, -90 at alpha=0),
# and similar for their western neighbours, the peaks should cluster at
# those quadrants (8 longitudes total).
# Round longitudes to nearest 45° to test.
top_lons = np.array([lon_target[np.unravel_index(ix, v_ll.shape)[1]]
                     for ix in idx])
print(f"\nPeak longitudes: {sorted(top_lons.tolist())}")
lon_quad = np.round(top_lons / 45.0) * 45.0
from collections import Counter
lon_hist = Counter(lon_quad.astype(int).tolist())
print(f"Peaks bucketed to nearest 45°: {dict(sorted(lon_hist.items()))}")
if all(abs(lq % 90) < 1 for lq in lon_hist.keys()):
    print("  → Peaks lie at longitude multiples of 90° — consistent with")
    print("    4-cube-corners-per-hemisphere structure at alpha=0.")
else:
    print("  → Peaks DO NOT cluster at 90° multiples — different structure.")
