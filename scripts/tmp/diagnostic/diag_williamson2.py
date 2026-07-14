"""Diagnostic script for Williamson case 2 v-wind analysis.

Runs Williamson 2 at C36 for 1 day with the current CDGrid model,
then prints detailed v-wind statistics and saves a v-wind visualization.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel,
    CDGridShallowWaterConfig,
)

# --- Setup ---
N = 36  # C36 resolution
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

dx_min = float(jnp.min(grid.dx))
config = CDGridShallowWaterConfig(
    A_h=1e4,
    hyperdiff_coeff=dx_min**4 / (86400.0 * 10),
)
model = CDGridShallowWaterModel(grid, config)

# --- Initial condition ---
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.unit.test_williamson2_cdgrid import williamson2_initial_condition
state0 = williamson2_initial_condition(cdgrid)
model.set_initial_mass(state0)

# --- Run 1 day ---
dt = 600.0  # 10 min
n_steps = int(86400 / dt)  # 1 day = 144 steps
print(f"Running Williamson 2 at C{N} for 1 day ({n_steps} steps, dt={dt}s)...")

state = state0
for i in range(n_steps):
    state = model.step(state, dt)
    if (i + 1) % 48 == 0:
        print(f"  Step {i+1}/{n_steps}")

# --- Diagnostics ---
print("\n=== Williamson 2 Diagnostics (1 day) ===\n")

# Height error
h_err = state.h - state0.h
area = cdgrid.base.area
total_area = float(jnp.sum(area))
h_range = float(jnp.max(state0.h) - jnp.min(state0.h))
l2_h = float(jnp.sqrt(jnp.sum(h_err**2 * area) / total_area))
linf_h = float(jnp.max(jnp.abs(h_err)))
print(f"Height error: L2={l2_h:.6f} m, Linf={linf_h:.6f} m")
print(f"Height error normalized: L2={l2_h/h_range:.6e}, Linf={linf_h/h_range:.6e}")

# V-wind analysis (v should be ~0 for Williamson 2)
# Convert D-grid corner winds to geographic (u_east, v_north)
ca = cdgrid.cos_angle_corner
sa = cdgrid.sin_angle_corner
u_geo = ca * state.u_d - sa * state.v_d  # eastward
v_geo = sa * state.u_d + ca * state.v_d  # northward

u_geo0 = ca * state0.u_d - sa * state0.v_d
v_geo0 = sa * state0.u_d + ca * state0.v_d

# V-wind error (v should be 0)
v_err = v_geo  # v_geo0 should be ~0 too, but numerical
v_err0 = v_geo0

print(f"\nInitial v-wind: max|v|={float(jnp.max(jnp.abs(v_geo0))):.6f} m/s")
print(f"Final v-wind:   max|v|={float(jnp.max(jnp.abs(v_geo))):.6f} m/s")
print(f"Final v-wind:   RMS(v)={float(jnp.sqrt(jnp.mean(v_geo**2))):.6f} m/s")

# U-wind error (u should be u_0*cos(lat))
u_0 = 38.61068276698372
lat_corner = cdgrid.lat_corner
u_exact = u_0 * jnp.cos(lat_corner)
u_err = u_geo - u_exact
print(f"\nU-wind error: max|du|={float(jnp.max(jnp.abs(u_err))):.6f} m/s")
print(f"U-wind error: RMS(du)={float(jnp.sqrt(jnp.mean(u_err**2))):.6f} m/s")

# Boundary vs interior v-wind
n = N
# Boundary corners: i=0, i=n, j=0, j=n
boundary_mask = jnp.zeros((6, n+1, n+1), dtype=bool)
boundary_mask = boundary_mask.at[:, 0, :].set(True)
boundary_mask = boundary_mask.at[:, n, :].set(True)
boundary_mask = boundary_mask.at[:, :, 0].set(True)
boundary_mask = boundary_mask.at[:, :, n].set(True)

v_bdy = jnp.abs(v_geo[boundary_mask])
v_int = jnp.abs(v_geo[~boundary_mask])
print(f"\nV-wind at boundary: max={float(jnp.max(v_bdy)):.4f}, RMS={float(jnp.sqrt(jnp.mean(v_bdy**2))):.4f} m/s")
print(f"V-wind at interior: max={float(jnp.max(v_int)):.4f}, RMS={float(jnp.sqrt(jnp.mean(v_int**2))):.4f} m/s")
print(f"Boundary/interior ratio: {float(jnp.sqrt(jnp.mean(v_bdy**2)) / jnp.sqrt(jnp.mean(v_int**2))):.2f}x")

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

    # Plot v-wind on each face
    for face in range(6):
        ax = axes[face // 3, face % 3]
        # Interpolate corners to cell centres for plotting
        v_cc = 0.25 * (v_geo[face, :-1, :-1] + v_geo[face, 1:, :-1]
                        + v_geo[face, :-1, 1:] + v_geo[face, 1:, 1:])
        im = ax.imshow(np.array(v_cc.T), origin='lower', cmap='RdBu_r',
                       vmin=-3, vmax=3)
        ax.set_title(f'Face {face}: v-wind [m/s]')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Williamson 2, C{N}, 1 day: v-wind (should be 0)', fontsize=14)
    plt.tight_layout()
    out_path = f'williamson2_vwind_C{N}_1day.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved v-wind plot to {out_path}")

    # Also plot u-wind error
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    for face in range(6):
        ax = axes2[face // 3, face % 3]
        u_cc = 0.25 * (u_err[face, :-1, :-1] + u_err[face, 1:, :-1]
                        + u_err[face, :-1, 1:] + u_err[face, 1:, 1:])
        im = ax.imshow(np.array(u_cc.T), origin='lower', cmap='RdBu_r',
                       vmin=-3, vmax=3)
        ax.set_title(f'Face {face}: u-wind error [m/s]')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Williamson 2, C{N}, 1 day: u-wind error', fontsize=14)
    plt.tight_layout()
    out_path2 = f'williamson2_uwind_err_C{N}_1day.png'
    plt.savefig(out_path2, dpi=150)
    print(f"Saved u-wind error plot to {out_path2}")

except ImportError:
    print("\nMatplotlib not available — skipping visualization")

print("\n=== Done ===")
