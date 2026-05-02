#!/usr/bin/env python
"""Edge-focused diagnostic for FV3-faithful cubed-sphere audit.

Runs Williamson 2 with both the production (fv3_sw_tendencies + RK3)
and the fixed d2a2c_vect duogrid path, generates native face plots.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm import constants

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'fv3_visual')
os.makedirs(OUT_DIR, exist_ok=True)

N = 24  # C24 for fast iteration
NSTEPS = 144  # 1 day at dt=600s
DT = 600.0
G = constants.g


def williamson2_ic(cdgrid):
    """Williamson case 2: steady-state geostrophic flow."""
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState,
    )
    grid = cdgrid.base
    n = grid.n
    u0 = 2 * jnp.pi * grid.radius / (12.0 * 86400.0)  # 12-day circumnavigation
    gh0 = 2.94e4  # mean geopotential height

    # Height field (geostrophic balance)
    h0 = gh0 / G
    omega = 7.292e-5
    h = h0 - (1.0 / G) * (grid.radius * omega * u0 + 0.5 * u0**2) * grid.sin_lat**2

    # Velocity: solid body rotation (geographic u=u0*cos(lat), v=0)
    u_east = u0 * grid.cos_lat
    v_north = jnp.zeros_like(u_east)

    # Rotate to grid-aligned
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)

    # Interpolate to D-grid edges
    u_d = 0.5 * (u_grid[:, :, :-1] + u_grid[:, :, 1:]) if u_grid.shape[2] == n else u_grid
    v_d = 0.5 * (v_grid[:, :-1, :] + v_grid[:, 1:, :]) if v_grid.shape[1] == n else v_grid

    # For edge-midpoint stagger: interpolate cell-centre to edges
    from legoesm.grids.halo import pad_halo
    u_pad = pad_halo(u_grid)
    v_pad = pad_halo(v_grid)
    u_d = 0.5 * (u_pad[:, 1:-1, :-1] + u_pad[:, 1:-1, 1:])  # (6, n, n+1)
    v_d = 0.5 * (v_pad[:, :-1, 1:-1] + v_pad[:, 1:, 1:-1])   # (6, n+1, n)

    return FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=jnp.zeros_like(h))


def run_williamson2(cdgrid, state0, label, use_csw=False):
    """Run Williamson 2 for NSTEPS and return final state."""
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        FV3EdgeShallowWaterModel,
    )
    config = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=0.0,
        use_experimental_csw=use_csw,
        boundary_fix=True,
    )
    model = FV3EdgeShallowWaterModel(cdgrid.base, config)
    model.set_initial_mass(state0)

    state = state0
    for i in range(NSTEPS):
        state = model.step(state, DT)
    return state


def plot_native_faces(field, title, fname, vmin=None, vmax=None, cmap='RdBu_r'):
    """Plot all 6 faces of a cubed-sphere field on native grid."""
    field_np = np.asarray(field)
    n = field_np.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for face in range(6):
        ax = axes[face // 3, face % 3]
        data = field_np[face]
        if vmin is None:
            _vmin, _vmax = data.min(), data.max()
        else:
            _vmin, _vmax = vmin, vmax
        im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=_vmin, vmax=_vmax)
        ax.set_title(f'Face {face}')
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)
    print(f"  Saved {fname}")


def edge_diagnostics(field, n, label):
    """Print edge/corner-focused diagnostics."""
    field_np = np.asarray(field)

    # Face boundary strips (1 cell from edge)
    west = field_np[:, 0, :]
    east = field_np[:, -1, :]
    south = field_np[:, :, 0]
    north = field_np[:, :, -1]

    # Interior (exclude 2 cells from each edge)
    if n > 4:
        interior = field_np[:, 2:-2, 2:-2]
    else:
        interior = field_np

    print(f"  {label}:")
    print(f"    Global: min={field_np.min():.6e} max={field_np.max():.6e} "
          f"mean={field_np.mean():.6e} std={field_np.std():.6e}")
    print(f"    Interior: min={interior.min():.6e} max={interior.max():.6e} "
          f"std={interior.std():.6e}")
    edge_all = np.concatenate([west.ravel(), east.ravel(), south.ravel(), north.ravel()])
    print(f"    Edges: min={edge_all.min():.6e} max={edge_all.max():.6e} "
          f"std={edge_all.std():.6e}")

    # Corner values
    corners = field_np[:, [0, 0, -1, -1], [0, -1, 0, -1]]
    print(f"    Corners: min={corners.min():.6e} max={corners.max():.6e} "
          f"std={corners.std():.6e}")

    # Edge/interior std ratio (artifact indicator)
    ratio = edge_all.std() / (interior.std() + 1e-30)
    print(f"    Edge/Interior std ratio: {ratio:.3f} "
          f"({'ARTIFACT' if ratio > 2.0 else 'OK'})")


def main():
    print(f"=== FV3 Edge Audit Diagnostic: Williamson 2 at C{N} ===")
    print(f"Steps: {NSTEPS}, dt: {DT}s")

    # ---- Production path (no duogrid) ----
    print("\n--- Production path (no duogrid, fv3_sw_tendencies + RK3) ---")
    grid_nodg = create_cubed_sphere(N)
    cdgrid_nodg = create_cubed_sphere_cdgrid(grid_nodg)
    ic_nodg = williamson2_ic(cdgrid_nodg)
    state_nodg = run_williamson2(cdgrid_nodg, ic_nodg, "nodg")

    # Error vs IC
    h_err_nodg = state_nodg.h - ic_nodg.h
    print(f"  h error: max_abs={float(jnp.max(jnp.abs(h_err_nodg))):.6e}")
    edge_diagnostics(h_err_nodg, N, "h_error (no duogrid)")

    # v-wind (should be ~0 for solid body rotation)
    u_cc_nodg = 0.5 * (state_nodg.u_d[:, :, :-1] + state_nodg.u_d[:, :, 1:])
    v_cc_nodg = 0.5 * (state_nodg.v_d[:, :-1, :] + state_nodg.v_d[:, 1:, :])
    from legoesm.grids.cubed_sphere import rotate_winds_grid_to_geo
    _, v_north_nodg = rotate_winds_grid_to_geo(u_cc_nodg, v_cc_nodg, grid_nodg.angle)
    edge_diagnostics(v_north_nodg, N, "v_north (no duogrid)")

    plot_native_faces(h_err_nodg, f"h error (no duogrid) C{N}, day 1", "w2_h_err_nodg.png")
    plot_native_faces(v_north_nodg, f"v_north (no duogrid) C{N}, day 1", "w2_vnorth_nodg.png")

    # ---- Duogrid path ----
    print("\n--- Duogrid path (fv3_sw_tendencies + RK3) ---")
    grid_dg = create_cubed_sphere(N, use_duogrid=True)
    cdgrid_dg = create_cubed_sphere_cdgrid(grid_dg)
    ic_dg = williamson2_ic(cdgrid_dg)
    state_dg = run_williamson2(cdgrid_dg, ic_dg, "dg")

    h_err_dg = state_dg.h - ic_dg.h
    print(f"  h error: max_abs={float(jnp.max(jnp.abs(h_err_dg))):.6e}")
    edge_diagnostics(h_err_dg, N, "h_error (duogrid)")

    u_cc_dg = 0.5 * (state_dg.u_d[:, :, :-1] + state_dg.u_d[:, :, 1:])
    v_cc_dg = 0.5 * (state_dg.v_d[:, :-1, :] + state_dg.v_d[:, 1:, :])
    _, v_north_dg = rotate_winds_grid_to_geo(u_cc_dg, v_cc_dg, grid_dg.angle)
    edge_diagnostics(v_north_dg, N, "v_north (duogrid)")

    plot_native_faces(h_err_dg, f"h error (duogrid) C{N}, day 1", "w2_h_err_dg.png")
    plot_native_faces(v_north_dg, f"v_north (duogrid) C{N}, day 1", "w2_vnorth_dg.png")

    # ---- Comparison ----
    print("\n--- Comparison ---")
    print(f"  h_err max (no dg): {float(jnp.max(jnp.abs(h_err_nodg))):.6e}")
    print(f"  h_err max (dg):    {float(jnp.max(jnp.abs(h_err_dg))):.6e}")
    print(f"  v_north max (no dg): {float(jnp.max(jnp.abs(v_north_nodg))):.6e}")
    print(f"  v_north max (dg):    {float(jnp.max(jnp.abs(v_north_dg))):.6e}")


if __name__ == "__main__":
    main()
