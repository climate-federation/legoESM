#!/usr/bin/env python
"""Probe the fold-adjacent cells to understand the blowup at j=329-331."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

GRID_FILE = Path("data/grids/eORCA1.2_mesh_mask.nc")


def main():
    geom = create_tripole_grid(str(GRID_FILE))

    print(f"Grid shape: {geom.n_lat} x {geom.n_lon}")
    print(f"Fold: fold_j={geom.fold.fold_j}, cap_j={geom.fold.cap_j}")
    print()

    # Raw metrics (BEFORE any clamping)
    lat_T = np.asarray(geom.lat_T) * 180 / np.pi
    lon_T = np.asarray(geom.lon_T) * 180 / np.pi
    dx_T = np.asarray(geom.dx_T)
    dy_T = np.asarray(geom.dy_T)
    area_T = np.asarray(geom.area_T)

    # Bathymetry / land mask
    import netCDF4
    ds = netCDF4.Dataset(str(GRID_FILE), "r")
    tmask = np.asarray(ds.variables["tmask"][0, 0])  # surface land mask
    mbathy = np.asarray(ds.variables["mbathy"][0])
    ds.close()

    print("=== Fold-adjacent rows (j=326 to j=331) ===")
    print(f"{'j':>3} {'lat_min':>8} {'lat_max':>8} {'dx_min':>10} {'dx_max':>10} "
          f"{'dy_min':>10} {'dy_max':>10} {'n_ocean':>8} {'n_total':>8}")
    for j in range(326, 332):
        print(f"{j:>3} {lat_T[j,:].min():>8.1f} {lat_T[j,:].max():>8.1f} "
              f"{dx_T[j,:].min():>10.1f} {dx_T[j,:].max():>10.1f} "
              f"{dy_T[j,:].min():>10.1f} {dy_T[j,:].max():>10.1f} "
              f"{int(tmask[j,:].sum()):>8} {tmask.shape[1]:>8}")

    print()

    # Focus on the blowup hotspot: around i=246, j=329-331
    print("=== Hotspot region (j=328-331, i=243-249) ===")
    print(f"{'j':>3} {'i':>3} {'lat':>7} {'lon':>8} {'dx_T':>10} {'dy_T':>10} "
          f"{'area_T':>12} {'ocean':>6} {'mbathy':>7}")
    for j in range(328, 332):
        for i in range(243, 250):
            print(f"{j:>3} {i:>3} {lat_T[j,i]:>7.1f} {lon_T[j,i]:>8.1f} "
                  f"{dx_T[j,i]:>10.1f} {dy_T[j,i]:>10.1f} "
                  f"{area_T[j,i]:>12.1f} {int(tmask[j,i]):>6} {int(mbathy[j,i]):>7}")

    print()

    # Check fold permutation at j=331
    fold = geom.fold
    perm_T = np.asarray(fold.perm_T)
    print(f"=== Fold permutation at j=331 (T-points) ===")
    print(f"perm_T[243:250] = {perm_T[243:250]}")
    print(f"(These map i → perm_T[i], reflecting the fold partner)")
    print()

    # Check: are the fold partner cells also land?
    print("=== Fold partner cells (j=331, partner j=331) ===")
    print(f"{'i':>3} {'partner_i':>10} {'ocean_src':>10} {'ocean_dst':>10} "
          f"{'dx_src':>10} {'dx_dst':>10}")
    for i in range(240, 255):
        pi = perm_T[i]
        print(f"{i:>3} {pi:>10} {int(tmask[331,i]):>10} {int(tmask[331,pi]):>10} "
              f"{dx_T[331,i]:>10.1f} {dx_T[331,pi]:>10.1f}")

    # Check rotation angles at the fold
    print()
    print("=== Rotation angles at fold (j=329-331, i=243-249) ===")
    cos_u = np.asarray(geom.cos_alpha_u)
    sin_u = np.asarray(geom.sin_alpha_u)
    cos_v = np.asarray(geom.cos_alpha_v)
    sin_v = np.asarray(geom.sin_alpha_v)
    print(f"{'j':>3} {'i':>3} {'cos_u':>10} {'sin_u':>10} {'cos_v':>10} {'sin_v':>10}")
    for j in range(329, 332):
        for i in range(243, 250):
            cu = cos_u[j, min(i, cos_u.shape[1]-1)]
            su = sin_u[j, min(i, sin_u.shape[1]-1)]
            cv = cos_v[min(j, cos_v.shape[0]-1), i]
            sv = sin_v[min(j, sin_v.shape[0]-1), i]
            print(f"{j:>3} {i:>3} {cu:>10.4f} {su:>10.4f} {cv:>10.4f} {sv:>10.4f}")


if __name__ == "__main__":
    main()
