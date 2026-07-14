#!/usr/bin/env python
"""Test the csw path (d2a2c_vect duogrid) on Williamson 2."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm import constants

N = 16  # small for speed
DT = 600.0
G = constants.g

print(f"=== Testing d2a2c_vect with duogrid on W2 at C{N} ===")

# Build grid with duogrid
grid = create_cubed_sphere(N, use_duogrid=True)
cdgrid = create_cubed_sphere_cdgrid(grid)
print(f"  Duogrid: {grid.duogrid is not None}")
print(f"  DuoGrid ng: {grid.duogrid.ng if grid.duogrid else 'N/A'}")

# Williamson 2 IC
u0 = 2 * jnp.pi * grid.radius / (12.0 * 86400.0)
gh0 = 2.94e4
h0 = gh0 / G
omega = constants.Omega
h = h0 - (1.0 / G) * (grid.radius * omega * u0 + 0.5 * u0**2) * grid.sin_lat**2
u_east = u0 * grid.cos_lat
v_north = jnp.zeros_like(u_east)

from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid
u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)

from legoesm.grids.halo import pad_halo
u_pad = pad_halo(u_grid)
v_pad = pad_halo(v_grid)
u_d = 0.5 * (u_pad[:, 1:-1, :-1] + u_pad[:, 1:-1, 1:])
v_d = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])
h_s = jnp.zeros_like(h)

# Test d2a2c_vect with duogrid
from legoesm.core.fv3_sw_core import d2a2c_vect
print("\n--- Testing d2a2c_vect ---")
ua, va, uc, vc, ut, vt = d2a2c_vect(u_d, v_d, cdgrid)

# Check that ua, va make sense (should be close to geographic wind)
from legoesm.grids.cubed_sphere import rotate_winds_grid_to_geo
u_geo, v_geo = rotate_winds_grid_to_geo(ua, va, grid.angle)
print(f"  ua max: {float(jnp.max(jnp.abs(ua))):.2f}")
print(f"  va max: {float(jnp.max(jnp.abs(va))):.2f}")
print(f"  u_geo should be ~{float(u0):.2f}: max={float(jnp.max(u_geo)):.2f} min={float(jnp.min(u_geo)):.2f}")
print(f"  v_geo should be ~0: max_abs={float(jnp.max(jnp.abs(v_geo))):.4f}")

# Check v_geo at edges vs interior
print(f"  v_geo interior std: {float(jnp.std(v_geo[:, 2:-2, 2:-2])):.6f}")
edge_v = jnp.concatenate([v_geo[:, 0, :].ravel(), v_geo[:, -1, :].ravel(),
                           v_geo[:, :, 0].ravel(), v_geo[:, :, -1].ravel()])
print(f"  v_geo edge std: {float(jnp.std(edge_v)):.6f}")
print(f"  Edge/Interior v_geo std ratio: {float(jnp.std(edge_v))/float(jnp.std(v_geo[:, 2:-2, 2:-2])+1e-30):.2f}")

# Test csw tendencies
print("\n--- Testing fv3_csw_tendencies ---")
from legoesm.core.fv3_sw_core import fv3_csw_tendencies
dh, du, dv = fv3_csw_tendencies(h, u_d, v_d, h_s, cdgrid, g=G)
print(f"  dh max_abs: {float(jnp.max(jnp.abs(dh))):.4e}")
print(f"  du max_abs: {float(jnp.max(jnp.abs(du))):.4e}")
print(f"  dv max_abs: {float(jnp.max(jnp.abs(dv))):.4e}")

# For W2 (steady state), tendencies should be near zero
print(f"  dh relative to h0: {float(jnp.max(jnp.abs(dh)))/h0:.4e}")
print(f"  du relative to u0: {float(jnp.max(jnp.abs(du)))/float(u0):.4e}")

# Run 10 steps
print("\n--- Running 10 steps with csw + RK3 ---")
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
)
state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
config = CDGridShallowWaterConfig(
    use_experimental_csw=True,
    hyperdiff_coeff=0.0,
    div_damp=0.0,
)
model = FV3EdgeShallowWaterModel(grid, config)
model.set_initial_mass(state)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(10):
        state = model.step(state, DT)
        if (i+1) % 5 == 0:
            h_err = float(jnp.max(jnp.abs(state.h - h)))
            print(f"  Step {i+1}: h_err_max={h_err:.4e}")

# Compare with production path
print("\n--- Running 10 steps with production path (fv3_sw_tendencies + RK3) ---")
state2 = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
config2 = CDGridShallowWaterConfig(
    use_experimental_csw=False,
    hyperdiff_coeff=0.0,
    div_damp=0.0,
)
model2 = FV3EdgeShallowWaterModel(grid, config2)
model2.set_initial_mass(state2)

for i in range(10):
    state2 = model2.step(state2, DT)
    if (i+1) % 5 == 0:
        h_err = float(jnp.max(jnp.abs(state2.h - h)))
        print(f"  Step {i+1}: h_err_max={h_err:.4e}")
