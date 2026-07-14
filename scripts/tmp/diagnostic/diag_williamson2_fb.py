"""Diagnostic: Williamson 2 with FV3 forward-backward model.

Compares the new FV3 FB model (no A-L gradient) against the baseline
CDGrid model.  Runs both at C36 for 1 day and reports v-wind errors.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig,
)

# --- Setup ---
N = 36
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

dx_min = float(jnp.min(grid.dx))
config = CDGridShallowWaterConfig(
    A_h=0.0,  # No Laplacian viscosity — let the scheme speak for itself
    hyperdiff_coeff=0.0,
    div_damp=0.0,
)
model = FV3FBShallowWaterModel(grid, config)

# --- Williamson 2 IC for edge-midpoint D-grid ---
g = constants.g
omega = constants.Omega
u_0 = 38.61068276698372
h_0 = 29400.0 / g
R = cdgrid.radius

# Height at cell centres
lat_c = cdgrid.base.lat  # (6, n, n)
h_init = h_0 - (R * omega * u_0 + 0.5 * u_0**2) * jnp.sin(lat_c)**2 / g

# u_d at edge-midpoint x positions: u = u_0 * cos(lat)
# In grid-aligned coords: u_grid = u_geo * cos(angle_edge)
lat_ux = cdgrid.lat_edge_x    # (6, n, n+1)
cos_ax = cdgrid.cos_angle_edge_x
sin_ax = cdgrid.sin_angle_edge_x
u_geo_x = u_0 * jnp.cos(lat_ux)
u_d = u_geo_x * cos_ax  # x-component of geographic wind at x-edge

# v_d at edge-midpoint y positions: v = 0 in geographic, but in grid-aligned:
# v_grid = -u_geo * sin(angle_edge) (from rotation)
lat_vy = cdgrid.lat_edge_y    # (6, n+1, n)
sin_ay = cdgrid.sin_angle_edge_y
cos_ay = cdgrid.cos_angle_edge_y
u_geo_y = u_0 * jnp.cos(lat_vy)
v_d = -u_geo_y * sin_ay  # y-component of geographic wind at y-edge

h_s = jnp.zeros_like(h_init)
state0 = FV3EdgeShallowWaterState(h=h_init, u_d=u_d, v_d=v_d, h_s=h_s)
model.set_initial_mass(state0)

# --- Check initial tendencies ---
print("Checking initial condition...")
print(f"  h range: {float(jnp.min(h_init)):.2f} to {float(jnp.max(h_init)):.2f} m")
print(f"  max|u_d|: {float(jnp.max(jnp.abs(u_d))):.4f} m/s")
print(f"  max|v_d|: {float(jnp.max(jnp.abs(v_d))):.4f} m/s")

# --- Run 1 step to test ---
dt = 600.0
print(f"\nRunning 1 step (dt={dt}s) to check stability...")
try:
    state1 = model.step(state0, dt)
    h_err_1 = float(jnp.max(jnp.abs(state1.h - state0.h)))
    print(f"  Step 1: max|dh|={h_err_1:.6f} m — {'STABLE' if jnp.all(jnp.isfinite(state1.h)) else 'UNSTABLE'}")
except Exception as e:
    print(f"  Step 1 FAILED: {e}")
    import traceback; traceback.print_exc()
    raise

# --- Run 1 day ---
n_steps = int(86400 / dt)
print(f"\nRunning Williamson 2 at C{N} for 1 day ({n_steps} steps)...")

state = state0
for i in range(n_steps):
    state = model.step(state, dt)
    if not jnp.all(jnp.isfinite(state.h)):
        print(f"  BLOWUP at step {i+1}!")
        break
    if (i + 1) % 48 == 0:
        dh = float(jnp.max(jnp.abs(state.h - state0.h)))
        print(f"  Step {i+1}/{n_steps}: max|dh|={dh:.4f}")

# --- Diagnostics ---
print("\n=== FV3 Forward-Backward Williamson 2 (1 day) ===\n")

# Height error
h_err = state.h - state0.h
area = cdgrid.base.area
total_area = float(jnp.sum(area))
h_range = float(jnp.max(state0.h) - jnp.min(state0.h))
l2_h = float(jnp.sqrt(jnp.sum(h_err**2 * area) / total_area))
linf_h = float(jnp.max(jnp.abs(h_err)))
print(f"Height error: L2={l2_h:.6f} m, Linf={linf_h:.6f} m")
print(f"Height error normalized: L2={l2_h/h_range:.6e}, Linf={linf_h/h_range:.6e}")

# Convert to geographic winds for analysis
cos_ax = cdgrid.cos_angle_edge_x
sin_ax = cdgrid.sin_angle_edge_x
# u_d is the grid-aligned x-component at x-edge midpoints
# Geographic eastward at x-edges: u_east = u_d * cos(angle) + ???
# Actually, at x-edge midpoints, u_d = u_geo * cos(angle), so u_geo = u_d / cos(angle)
# But this is ill-conditioned near cos(angle)=0. Use the full rotation.
# u_east = u_d / cos(angle) when v_geo = 0

# For Williamson 2, v should be 0. The v_d field should be -u_0*cos(lat)*sin(angle).
# If v_geo = 0: v_d = -u_geo * sin(angle) → u_geo = -v_d / sin(angle)

# Let's measure the error in the most direct way:
# The exact v_d is -u_0 * cos(lat_edge_y) * sin(angle_edge_y)
v_d_exact = -u_0 * jnp.cos(cdgrid.lat_edge_y) * cdgrid.sin_angle_edge_y
v_err = state.v_d - v_d_exact
print(f"\nv_d error: max|err|={float(jnp.max(jnp.abs(v_err))):.6f} m/s")
print(f"v_d error: RMS={float(jnp.sqrt(jnp.mean(v_err**2))):.6f} m/s")

u_d_exact = u_0 * jnp.cos(cdgrid.lat_edge_x) * cdgrid.cos_angle_edge_x
u_err = state.u_d - u_d_exact
print(f"u_d error: max|err|={float(jnp.max(jnp.abs(u_err))):.6f} m/s")
print(f"u_d error: RMS={float(jnp.sqrt(jnp.mean(u_err**2))):.6f} m/s")

# Mass conservation
mass_0 = float(jnp.sum(state0.h * area))
mass_f = float(jnp.sum(state.h * area))
print(f"\nMass conservation: rel_err={abs(mass_f - mass_0) / abs(mass_0):.2e}")

# --- Visualization ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for face in range(6):
        ax = axes[face // 3, face % 3]
        # v_d error at y-edges: shape (n+1, n), plot as is
        im = ax.imshow(np.array(v_err[face].T), origin='lower', cmap='RdBu_r',
                       vmin=-3, vmax=3)
        ax.set_title(f'Face {face}: v_d error [m/s]')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'FV3-FB Williamson 2, C{N}, 1 day: v_d error', fontsize=14)
    plt.tight_layout()
    out_path = f'williamson2_fb_vd_err_C{N}_1day.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
except ImportError:
    print("\nMatplotlib not available")

print("\n=== Done ===")
