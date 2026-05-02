"""Iter-987: probe pad_halo halo values at face=1 west cube-edge seam.

Compares nearest-index copy (kinked positions) vs cube_rmp Lagrange
remap (extended positions) for cell-centred zeta_abs at j=21
(lat ~6.34°, the day-1 v_ll_Linf peak location).

Finding: Fortran's `cube_rmp` (`fv_duogrid.F90:977-1137`) applies
the same k2e_coef weighted Lagrange remap as our Python.  Skipping
cube_rmp on FB chain C36 W2 1-day yields only 0.2% improvement on
v_ll_Linf (55.61 → 55.52) — not the dominant lever for the seam
artifact.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.duogrid import (
    cube_rmp_vectorized,
    fill_corner_region,
)
from legoesm.grids.halo import _pad_halo_local_h2, pad_halo
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def main():
    N = 36
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)

    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    dx_u = np.asarray(cdgrid.dx_edge_y)
    dy_v = np.asarray(cdgrid.dy_edge_x)
    vt_circ = np.asarray(u_d) * dx_u
    ut_circ = np.asarray(v_d) * dy_v
    rarea = 1.0 / np.asarray(cdgrid.base.area)
    zeta = rarea * (vt_circ[:, :, :-1] - vt_circ[:, :, 1:]
                     + ut_circ[:, 1:, :] - ut_circ[:, :-1, :])
    zeta_abs = zeta + np.asarray(cdgrid.base.f)

    # Step 1: nearest-index copy only
    zeta_pad_nearest = np.asarray(
        _pad_halo_local_h2(jnp.asarray(zeta_abs), None))

    # Step 2: cube_rmp + fill_corners
    zeta_pad_cube_rmp = np.asarray(
        cube_rmp_vectorized(
            jnp.asarray(zeta_pad_nearest), grid.duogrid, 2))
    zeta_pad_fill = np.asarray(
        fill_corner_region(
            jnp.asarray(zeta_pad_cube_rmp), grid.duogrid, 2))

    print("=== Iter-987 pad_halo halo values at face=1 west j=21 ===")
    print(f"  i_pad | nearest    | after cube_rmp | after fill_corner")
    for i_pad in [0, 1, 2, 3]:
        n_val = zeta_pad_nearest[1, i_pad, 23]
        c_val = zeta_pad_cube_rmp[1, i_pad, 23]
        f_val = zeta_pad_fill[1, i_pad, 23]
        print(f"  {i_pad}     | {n_val:9.4e} | {c_val:14.4e} | "
              f"{f_val:17.4e}")

    print("\nFace 0 east interior (the kinked-position 'expected'):")
    for i in [33, 34, 35]:
        print(f"  zeta_abs[face=0, i={i}, j=21] = "
              f"{zeta_abs[0, i, 21]:.4e}")

    print("\nFace 1 interior (sequence into the face):")
    for i in [0, 1, 2, 3]:
        print(f"  zeta_abs[face=1, i={i}, j=21] = "
              f"{zeta_abs[1, i, 21]:.4e}")


if __name__ == "__main__":
    main()
