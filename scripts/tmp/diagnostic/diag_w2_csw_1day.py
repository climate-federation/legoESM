#!/usr/bin/env python
"""Test csw+duogrid path for 1 day at C24."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import warnings
from legoesm.grids.cubed_sphere import create_cubed_sphere, rotate_winds_geo_to_grid, rotate_winds_grid_to_geo
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm import constants
import numpy as np

N = 24
DT = 600.0
G = constants.g
NSTEPS = 144  # 1 day

print(f"=== Williamson 2 at C{N}, 1 day ({NSTEPS} steps) ===")

for label, use_dg, use_csw in [
    ("production (no dg)", False, False),
    ("production (dg)", True, False),
    ("csw+dg", True, True),
]:
    grid = create_cubed_sphere(N, use_duogrid=use_dg)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    u0 = 2 * jnp.pi * grid.radius / (12.0 * 86400.0)
    h0 = 2.94e4 / G
    omega = constants.Omega
    h = h0 - (1.0 / G) * (grid.radius * omega * u0 + 0.5 * u0**2) * grid.sin_lat**2
    u_east = u0 * grid.cos_lat
    v_north = jnp.zeros_like(u_east)
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)
    u_pad = pad_halo(u_grid)
    v_pad = pad_halo(v_grid)
    u_d = 0.5 * (u_pad[:, 1:-1, :-1] + u_pad[:, 1:-1, 1:])
    v_d = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])
    h_s = jnp.zeros_like(h)
    state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)

    config = CDGridShallowWaterConfig(use_experimental_csw=use_csw, boundary_fix=True)
    model = FV3EdgeShallowWaterModel(grid, config)
    model.set_initial_mass(state)

    print(f"\n--- {label} ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(NSTEPS):
            state = model.step(state, DT)
            if (i + 1) % 48 == 0:
                h_err = float(jnp.max(jnp.abs(state.h - h)))
                u_cc = 0.5 * (state.u_d[:, :, :-1] + state.u_d[:, :, 1:])
                v_cc = 0.5 * (state.v_d[:, :-1, :] + state.v_d[:, 1:, :])
                _, vn = rotate_winds_grid_to_geo(u_cc, v_cc, grid.angle)
                vn_np = np.asarray(vn)
                int_std = vn_np[:, 2:-2, 2:-2].std()
                edge = np.concatenate([vn_np[:, 0, :].ravel(), vn_np[:, -1, :].ravel(),
                                       vn_np[:, :, 0].ravel(), vn_np[:, :, -1].ravel()])
                edge_std = edge.std()
                print(f"  Step {i+1}: h_err={h_err:.1f}, v_north_edge/int={edge_std/(int_std+1e-30):.1f}")

    h_err_final = float(jnp.max(jnp.abs(state.h - h)))
    print(f"  Final h_err: {h_err_final:.1f}")
