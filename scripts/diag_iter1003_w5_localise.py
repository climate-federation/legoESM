"""Iter-1003: localise W5 day-5 instability on production CDGrid path.

Result: W5 is artifact-free through day 4 (speeds <40 m/s,
physically reasonable for zonal flow over mountain).  Day 5
instability appears at face=4 (north pole) i=0 j=34 — cube-edge
on the polar face.

Same cube-edge mechanism as W2 FB-chain seam mode (iter-985)
but on the polar face.  Structural fix needed for W5 long-term
stability.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test5,
)
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import _div_damp_cube


def main():
    N = 36
    DT = 300.0
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test5(grid)
    u0 = 20.0
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=12.0 * _div_damp_cube(N),
        boundary_fix=True, damp_v=0.04, nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)

    print("=== Iter-1003 W5 day-by-day localisation ===")
    print(f"{'day':>3} | speed_max (m/s) | location")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for day in range(1, 8):
            for _ in range(int(86400 / DT)):
                state = model.step(state, DT)
            u = np.asarray(state.u_d)
            v = np.asarray(state.v_d)
            u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
            speed = np.sqrt(u_cc**2 + v_cc**2)
            speed_max = speed.max()
            idx = np.unravel_index(np.argmax(speed), speed.shape)
            f, i, j = idx
            is_corner = (i in [0, N - 1]) and (j in [0, N - 1])
            is_edge = (i in [0, N - 1] or j in [0, N - 1])
            loc = "CORNER" if is_corner else (
                "EDGE" if is_edge else "INTERIOR"
            )
            lat_d = float(np.degrees(grid.lat[f, i, j]))
            lon_d = float(np.degrees(grid.lon[f, i, j]))
            print(f"{day:>3d} | {speed_max:>15.2f} | "
                  f"face={f} i={i} j={j} ({loc}) "
                  f"lat={lat_d:.1f}° lon={lon_d:.1f}°")


if __name__ == "__main__":
    main()
