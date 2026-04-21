"""Iter-762 diagnostic: localise the remaining W2 v-wind artifact
peaks at the current Fortran-faithful matrix config (iter-761).

Baseline v_ll_Linf is 0.159 m/s with damp_v=0.06, nord_v=2,
div_damp=8*_div_damp_cube(n), hyperdiff_coeff=0.  Visual inspection
shows weak polar bands + mid-latitude cube-corner seams.  This
diagnostic reports the TOP peaks in regridded v_ll to determine
whether they are predominantly:
  (a) cube-corner (lat ±35°) — mode A from iter-745,
  (b) polar (lat ±86°) — mode B from iter-745,
  (c) a mix with specific latitude/longitude structure that
      points to a newly-dominant mechanism.
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
from scripts.run_atmosphere_test_matrix import _div_damp_cube, _regrid_2d


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

dt = 300.0
config = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=8.0 * _div_damp_cube(n),
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config)
cdgrid = model.cdgrid

sw = williamson_test2(grid)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
model.set_initial_mass(state)

for _ in range(int(86400/dt)):
    state = model.step(state, dt)

ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
ca = np.asarray(ca_4edge); sa = np.asarray(sa_4edge)
u_cc = 0.5*(np.asarray(state.u_d)[:,:,:-1] + np.asarray(state.u_d)[:,:,1:])
v_cc = 0.5*(np.asarray(state.v_d)[:,:-1,:] + np.asarray(state.v_d)[:,1:,:])
v_north = sa * u_cc + ca * v_cc

lon_deg = np.asarray(grid.lon)*180/np.pi
lat_deg = np.asarray(grid.lat)*180/np.pi
v_ll = _regrid_2d(v_north, lon_deg, lat_deg, "cube")

print(f"Iter-761 config v_ll: max|v_ll|={np.max(np.abs(v_ll)):.3e} m/s")
print(f"Iter-761 config v_north_face: max|v_north|={np.max(np.abs(v_north)):.3e} m/s\n")

# Per-face Linf
print("Per-face |v_north| Linf:")
for f in range(6):
    fl = float(np.max(np.abs(v_north[f])))
    argmax = np.argmax(np.abs(v_north[f]))
    i, j = np.unravel_index(argmax, v_north[f].shape)
    print(f"  face{f}: {fl:.3e} at (i={i},j={j}) "
          f"lat={lat_deg[f,i,j]:+.1f}° lon={lon_deg[f,i,j]:+.1f}°")

# Top-12 peaks in lat-lon
abs_v = np.abs(v_ll)
top_k = 12
idx = np.argsort(abs_v.flatten())[-top_k:][::-1]
lat_target = np.linspace(-90, 90, v_ll.shape[0])
lon_target = np.linspace(-180, 180, v_ll.shape[1], endpoint=False)
print(f"\nTop {top_k} peaks in lat-lon regridded:")
print(f"  {'rank':>4}  {'lat':>7}  {'lon':>8}  {'|v_ll|':>10}  {'sign':>4}")
for k, ix in enumerate(idx):
    i, j = np.unravel_index(ix, v_ll.shape)
    sign = 1 if v_ll[i,j] > 0 else -1
    print(f"  {k+1:>4}  {lat_target[i]:+7.2f}  {lon_target[j]:+8.2f}  "
          f"{abs_v.flatten()[ix]:>10.3e}  {sign:>4}")

# Latitude distribution of peaks
top_lats = np.array([lat_target[np.unravel_index(ix, v_ll.shape)[0]]
                     for ix in idx])
cube_lat = np.degrees(np.arctan(1.0 / np.sqrt(2.0)))
print(f"\nMean |peak lat|: {np.mean(np.abs(top_lats)):.2f}° "
      f"(cube corner ±{cube_lat:.2f}°, polar ±86°)")

near_cube = np.sum(np.abs(np.abs(top_lats) - cube_lat) < 5.0)
near_polar = np.sum(np.abs(np.abs(top_lats) - 86.0) < 5.0)
print(f"Peaks within 5° of cube-corner ±{cube_lat:.0f}°: {near_cube}/{top_k}")
print(f"Peaks within 5° of polar ±86°: {near_polar}/{top_k}")

# Longitude distribution (mod 90°)
top_lons = np.array([lon_target[np.unravel_index(ix, v_ll.shape)[1]]
                     for ix in idx])
lon_mod90 = np.abs(top_lons) % 90.0
# Prefer smaller between lon_mod90 and 90 - lon_mod90
lon_dist_cube = np.minimum(lon_mod90, 90.0 - lon_mod90)
print(f"\nMean longitude distance-to-nearest-45° axis: "
      f"{np.mean(lon_dist_cube):.2f}° "
      f"(cube corners at 45° lon multiples, mid-edge midpoints at 0° or 90°)")
