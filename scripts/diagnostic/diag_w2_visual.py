#!/usr/bin/env python
"""Visual diagnostic: native face plots for Williamson 2."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.cubed_sphere import create_cubed_sphere, rotate_winds_geo_to_grid, rotate_winds_grid_to_geo
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
)
from legoesm import constants

OUT = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'fv3_visual')
os.makedirs(OUT, exist_ok=True)

N = 24
DT = 600.0
G = constants.g
NSTEPS = 144

def make_state(grid):
    u0 = 2*jnp.pi*grid.radius/(12*86400)
    h0 = 2.94e4/G
    omega = 7.292e-5
    h = h0 - (1/G)*(grid.radius*omega*u0 + 0.5*u0**2)*grid.sin_lat**2
    u_east = u0*grid.cos_lat
    v_north = jnp.zeros_like(u_east)
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)
    u_pad = pad_halo(u_grid); v_pad = pad_halo(v_grid)
    u_d = 0.5*(u_pad[:,1:-1,:-1]+u_pad[:,1:-1,1:])
    v_d = 0.5*(v_pad[:,:-1,1:-1]+v_pad[:,1:,1:-1])
    return FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=jnp.zeros_like(h)), h

def run_sim(grid, cdgrid, state0, use_csw):
    config = CDGridShallowWaterConfig(use_experimental_csw=use_csw, boundary_fix=True)
    model = FV3EdgeShallowWaterModel(grid, config)
    model.set_initial_mass(state0)
    state = state0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(NSTEPS):
            state = model.step(state, DT)
    return state

def plot_6faces(field_np, title, fname, cmap='RdBu_r', sym=True):
    n = field_np.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if sym:
        vmax = np.abs(field_np).max()
        vmin = -vmax
    else:
        vmin, vmax = field_np.min(), field_np.max()
    for f in range(6):
        ax = axes[f//3, f%3]
        im = ax.imshow(field_np[f].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'Face {f}')
        # Mark edges
        for x in [0, n-1]:
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
            ax.axhline(x, color='gray', linewidth=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax, shrink=0.7, format='%.0f')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

print(f"=== W2 Visual Diagnostic at C{N}, 1 day ===")

# Production (no duogrid)
grid_p = create_cubed_sphere(N)
cdgrid_p = create_cubed_sphere_cdgrid(grid_p)
state0_p, h0_p = make_state(grid_p)
state_p = run_sim(grid_p, cdgrid_p, state0_p, use_csw=False)

h_err_p = np.asarray(state_p.h - h0_p)
u_cc_p = 0.5*(state_p.u_d[:,:,:-1]+state_p.u_d[:,:,1:])
v_cc_p = 0.5*(state_p.v_d[:,:-1,:]+state_p.v_d[:,1:,:])
_, vn_p = rotate_winds_grid_to_geo(u_cc_p, v_cc_p, grid_p.angle)
vn_p = np.asarray(vn_p)

# csw+duogrid
grid_d = create_cubed_sphere(N, use_duogrid=True)
cdgrid_d = create_cubed_sphere_cdgrid(grid_d)
state0_d, h0_d = make_state(grid_d)
state_d = run_sim(grid_d, cdgrid_d, state0_d, use_csw=True)

h_err_d = np.asarray(state_d.h - h0_d)
u_cc_d = 0.5*(state_d.u_d[:,:,:-1]+state_d.u_d[:,:,1:])
v_cc_d = 0.5*(state_d.v_d[:,:-1,:]+state_d.v_d[:,1:,:])
_, vn_d = rotate_winds_grid_to_geo(u_cc_d, v_cc_d, grid_d.angle)
vn_d = np.asarray(vn_d)

print(f"\nProduction: h_err max={np.abs(h_err_p).max():.0f}, v_north max={np.abs(vn_p).max():.1f}")
print(f"csw+dg:    h_err max={np.abs(h_err_d).max():.0f}, v_north max={np.abs(vn_d).max():.1f}")

plot_6faces(h_err_p, f'h error: Production C{N} day 1', 'w2_herr_production.png')
plot_6faces(h_err_d, f'h error: csw+duogrid C{N} day 1', 'w2_herr_csw_dg.png')
plot_6faces(vn_p, f'v_north: Production C{N} day 1', 'w2_vnorth_production.png')
plot_6faces(vn_d, f'v_north: csw+duogrid C{N} day 1', 'w2_vnorth_csw_dg.png')

# Difference plot (improvement)
h_diff = np.abs(h_err_d) - np.abs(h_err_p)
plot_6faces(h_diff, f'|h_err| improvement (csw+dg - prod) C{N}', 'w2_herr_improvement.png')

print(f"\nImprovement: h_err max reduced by {(1-np.abs(h_err_d).max()/np.abs(h_err_p).max())*100:.0f}%")
print(f"  Production edge/int ratio: {vn_p[:,[0,-1],:].std() / (vn_p[:,2:-2,2:-2].std()+1e-30):.2f}")
print(f"  csw+dg edge/int ratio: {vn_d[:,[0,-1],:].std() / (vn_d[:,2:-2,2:-2].std()+1e-30):.2f}")
