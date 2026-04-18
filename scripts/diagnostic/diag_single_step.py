#!/usr/bin/env python
"""Single-step tendency comparison: duogrid vs plain."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere, rotate_winds_geo_to_grid
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm import constants

N = 24
G = constants.g

def make_ic(grid, cdgrid):
    u0 = 2 * jnp.pi * grid.radius / (12.0 * 86400.0)
    gh0 = 2.94e4
    h0 = gh0 / G
    omega = 7.292e-5
    h = h0 - (1.0 / G) * (grid.radius * omega * u0 + 0.5 * u0**2) * grid.sin_lat**2
    u_east = u0 * grid.cos_lat
    v_north = jnp.zeros_like(u_east)
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)
    u_pad = pad_halo(u_grid)
    v_pad = pad_halo(v_grid)
    u_d = 0.5 * (u_pad[:, 1:-1, :-1] + u_pad[:, 1:-1, 1:])
    v_d = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])
    h_s = jnp.zeros_like(h)
    return h, u_d, v_d, h_s

def analyze(dh, du, dv, label):
    dh_np = np.asarray(dh)
    du_np = np.asarray(du)
    # Interior vs edge
    int_dh = dh_np[:, 2:-2, 2:-2]
    edge_dh = np.concatenate([dh_np[:, 0, :].ravel(), dh_np[:, -1, :].ravel(),
                               dh_np[:, :, 0].ravel(), dh_np[:, :, -1].ravel()])
    print(f"  {label}:")
    print(f"    dh: max_abs={np.abs(dh_np).max():.6e}, interior_std={int_dh.std():.6e}, "
          f"edge_std={edge_dh.std():.6e}, ratio={edge_dh.std()/(int_dh.std()+1e-30):.2f}")
    int_du = du_np[:, 2:-2, 2:-2]
    edge_du = np.concatenate([du_np[:, 0, :].ravel(), du_np[:, -1, :].ravel()])
    print(f"    du: max_abs={np.abs(du_np).max():.6e}, interior_std={int_du.std():.6e}, "
          f"edge_std={edge_du.std():.6e}, ratio={edge_du.std()/(int_du.std()+1e-30):.2f}")

print(f"=== Single-step W2 tendencies at C{N} ===\n")

# Plain (no duogrid)
grid_p = create_cubed_sphere(N)
cdgrid_p = create_cubed_sphere_cdgrid(grid_p)
h, u_d, v_d, h_s = make_ic(grid_p, cdgrid_p)
dh_p, du_p, dv_p = fv3_sw_tendencies(h, u_d, v_d, h_s, cdgrid_p, g=G)
analyze(dh_p, du_p, dv_p, "no duogrid")

# Duogrid
grid_d = create_cubed_sphere(N, use_duogrid=True)
cdgrid_d = create_cubed_sphere_cdgrid(grid_d)
h2, u_d2, v_d2, h_s2 = make_ic(grid_d, cdgrid_d)
dh_d, du_d2, dv_d = fv3_sw_tendencies(h2, u_d2, v_d2, h_s2, cdgrid_d, g=G)
analyze(dh_d, du_d2, dv_d, "duogrid")

# Compare tendencies
print(f"\n--- Tendency difference (dg - plain) ---")
diff_dh = np.asarray(dh_d - dh_p)
diff_du = np.asarray(du_d2 - du_p)
print(f"  dh diff: max_abs={np.abs(diff_dh).max():.6e}")
print(f"  du diff: max_abs={np.abs(diff_du).max():.6e}")
# Where are the largest differences?
face, i, j = np.unravel_index(np.argmax(np.abs(diff_dh)), diff_dh.shape)
print(f"  dh max diff at face={face}, i={i}, j={j} (n={N})")
face2, i2, j2 = np.unravel_index(np.argmax(np.abs(diff_du)), diff_du.shape)
print(f"  du max diff at face={face2}, i={i2}, j={j2}")

# For W2 steady state, all tendencies should ideally be zero.
# The tendencies represent the numerical error of the discretization.
print(f"\n--- Ideal: tendencies = 0 for W2 ---")
print(f"  no-dg dh L2: {float(jnp.sqrt(jnp.mean(dh_p**2))):.6e}")
print(f"  dg    dh L2: {float(jnp.sqrt(jnp.mean(dh_d**2))):.6e}")
print(f"  no-dg du L2: {float(jnp.sqrt(jnp.mean(du_p**2))):.6e}")
print(f"  dg    du L2: {float(jnp.sqrt(jnp.mean(du_d2**2))):.6e}")
