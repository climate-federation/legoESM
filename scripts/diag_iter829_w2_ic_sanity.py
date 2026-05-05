"""Iter-829 diagnostic: W2 analytical IC v_north should be zero.

Sanity check: for the W2 alpha=0 analytical IC (solid-body rotation
about the polar axis), v_north is exactly 0 everywhere.  Any non-
zero v_north at t=0 would indicate an IC projection issue that
could contribute to the mode-A residual independently of the
dynamical cube-vertex mechanism.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist_to_nearest_cube_vertex_deg(lat_deg, lon_deg):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
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


for n in (24, 36, 48):
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    model = FV3EdgeShallowWaterModel(grid)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    u_cc = 0.5 * (np.asarray(u_d)[:, :, :-1] + np.asarray(u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(v_d)[:, :-1, :] + np.asarray(v_d)[:, 1:, :])
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    vn_peak = float(np.max(np.abs(v_north)))

    idx = np.unravel_index(np.argmax(np.abs(v_north)), v_north.shape)
    lat_cc = np.rad2deg(np.asarray(grid.lat))
    lon_cc = np.rad2deg(np.asarray(grid.lon))
    lon_cc = np.where(lon_cc > 180.0, lon_cc - 360.0, lon_cc)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    latp = float(lat_cc[face, ci, cj])
    lonp = float(lon_cc[face, ci, cj])
    gc = _gc_dist_to_nearest_cube_vertex_deg(latp, lonp)

    print(f"C{n} W2 IC v_north: peak |v_north| = {vn_peak:.3e} m/s "
          f"at face {face} ({ci},{cj}) lat={latp:+.2f}° "
          f"lon={lonp:+.2f}° GC-to-vertex={gc:.2f}°")

print()
print("Interpretation cues:")
print("- Peak |v_north| should be ~machine-precision (<1e-12) for a")
print("  correct IC projection — analytical v_north = 0.")
print("- Any non-zero IC v_north indicates the IC construction has a")
print("  projection error that contributes to the W2 mode-A independent")
print("  of dynamics.")
