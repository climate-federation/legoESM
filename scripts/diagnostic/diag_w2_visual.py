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

# Iter-820b cleanup: removed `rotate_winds_geo_to_grid`,
# `rotate_winds_grid_to_geo`, and `pad_halo` imports — all unused
# after iter-820 switched to `williamson_test2(grid)` +
# `cell_centre_angles_from_4edge(cdgrid)`.
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
)
from legoesm import constants

OUT = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'fv3_visual')
os.makedirs(OUT, exist_ok=True)

N = 36
# Iter-797: match the production matrix dt (run_atmosphere_test_matrix.py
# line 1177) which uses dt=300s for W2/W5.  dt=600s (prior value) with
# the iter-761 canonical config was numerically unstable at C24.
# Iter-813: bumped N to 36 to match the production W2 sentinel resolution
# (test_w2_alpha0_c36_1day_iter761_matrix_config uses n=36).
DT = 300.0
G = constants.g
NSTEPS = int(86400 / DT)  # 1 day

def make_state(grid, cdgrid):
    # Iter-820: use williamson_test2(grid) for h exactly as the
    # production W2 sentinel does (tests/unit/test_cdgrid_fv3_
    # regression.py:3929 → `sw = williamson_test2(grid)`).  Prior
    # iter-797 used an inline analytical h formula which differed
    # from williamson_test2's h by ~2.4× on the post-regrid v_ll_
    # Linf metric (0.382 vs 0.159 iter-787 baseline).
    #
    # u_d, v_d at edge midpoints using analytical cos_angle_edge_*
    # and lat_edge_* is accurate to machine precision; matches the
    # production sentinel exactly.
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2)
    sw = williamson_test2(grid)
    u0 = 2*jnp.pi*grid.radius/(12*86400)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data), sw.h.data

def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def run_sim(grid, cdgrid, state0, use_csw, n=N):
    # Iter-921: match the CURRENT iter-893 production matrix config
    # (run_atmosphere_test_matrix.py + test_w2_alpha0_c36_1day_iter761
    # _matrix_config), which adds `apply_fortran_xppm_boundary=True`
    # to the iter-761 baseline.  This reduces v_ll_Linf from 0.159
    # (iter-820 baseline) to 0.132 m/s.  Without this flag, the visual
    # snapshot reflects the stale iter-820 state, not the live
    # production sentinel.  iter-797/iter-761: hyperdiff=0, div_damp=
    # 8*base, damp_v=0.06, nord_v=2, boundary_fix=True; iter-893:
    # apply_fortran_xppm_boundary=True (PPM cube-edge alignment fix).
    config = CDGridShallowWaterConfig(
        use_experimental_csw=use_csw,
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True)
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
state0_p, h0_p = make_state(grid_p, cdgrid_p)
state_p = run_sim(grid_p, cdgrid_p, state0_p, use_csw=False)

h_err_p = np.asarray(state_p.h - h0_p)
# Iter-820: use cell_centre_angles_from_4edge(cdgrid) for v_north
# projection exactly as the production W2 sentinel does (tests/
# unit/test_cdgrid_fv3_regression.py:4157).  Prior use of
# `grid.angle` / `rotate_winds_grid_to_geo` gave a ~2.4x larger
# v_ll_Linf number (0.382 vs iter-787 baseline 0.159) because the
# 4-edge-angle convention is what the sentinel computes against.
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
ca_4edge_p, sa_4edge_p = cell_centre_angles_from_4edge(cdgrid_p)
u_cc_p = 0.5*(np.asarray(state_p.u_d)[:,:,:-1]+np.asarray(state_p.u_d)[:,:,1:])
v_cc_p = 0.5*(np.asarray(state_p.v_d)[:,:-1,:]+np.asarray(state_p.v_d)[:,1:,:])
vn_p = np.asarray(sa_4edge_p) * u_cc_p + np.asarray(ca_4edge_p) * v_cc_p

# Iter-812: compare RK3+duogrid (post iter-808 sign-flip sync fix) —
# NOT the old csw+duogrid path (use_experimental_csw=True), which goes
# through fv3_csw_tendencies and is known unstable.  The iter-808 fix
# is in synchronize_cgrid_fluxes, exercised by fv3_sw_tendencies, which
# is what FV3EdgeShallowWaterModel.step calls with use_experimental_csw=
# False (default).  So we flip use_csw=False and use_duogrid=True.
grid_d = create_cubed_sphere(N, use_duogrid=True)
cdgrid_d = create_cubed_sphere_cdgrid(grid_d)
state0_d, h0_d = make_state(grid_d, cdgrid_d)
state_d = run_sim(grid_d, cdgrid_d, state0_d, use_csw=False)

h_err_d = np.asarray(state_d.h - h0_d)
# Iter-820: use cell_centre_angles_from_4edge(cdgrid) per sentinel.
ca_4edge_d, sa_4edge_d = cell_centre_angles_from_4edge(cdgrid_d)
u_cc_d = 0.5*(np.asarray(state_d.u_d)[:,:,:-1]+np.asarray(state_d.u_d)[:,:,1:])
v_cc_d = 0.5*(np.asarray(state_d.v_d)[:,:-1,:]+np.asarray(state_d.v_d)[:,1:,:])
vn_d = np.asarray(sa_4edge_d) * u_cc_d + np.asarray(ca_4edge_d) * v_cc_d

print(f"\nProduction (LEGACY): h_err max={np.abs(h_err_p).max():.2f}, v_north max={np.abs(vn_p).max():.2f}")
print(f"DUOGRID (RK3 post-iter-808): h_err max={np.abs(h_err_d).max():.2f}, v_north max={np.abs(vn_d).max():.2f}")

plot_6faces(h_err_p, f'h error: Production LEGACY C{N} day 1', 'w2_herr_production.png')
plot_6faces(h_err_d, f'h error: DUOGRID RK3 C{N} day 1', 'w2_herr_csw_dg.png')
plot_6faces(vn_p, f'v_north: Production LEGACY C{N} day 1', 'w2_vnorth_production.png')
plot_6faces(vn_d, f'v_north: DUOGRID RK3 C{N} day 1', 'w2_vnorth_csw_dg.png')

# Difference plot (improvement)
h_diff = np.abs(h_err_d) - np.abs(h_err_p)
plot_6faces(h_diff, f'|h_err| DUOGRID - LEGACY C{N}', 'w2_herr_improvement.png')

# Iter-814: add lat-lon regridded v_north plots to separate the
# physical signal from native-grid visualization artifacts (face 4's
# X-pattern at the geographic pole is a coordinate-rotation quirk,
# not a physical artifact).  After regridding to lat-lon, the pole
# is represented cleanly as a single point.
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
vn_p_ll = apply_cubedsphere_to_latlon(vn_p, weights)
vn_d_ll = apply_cubedsphere_to_latlon(vn_d, weights)
h_err_p_ll = apply_cubedsphere_to_latlon(h_err_p, weights)
h_err_d_ll = apply_cubedsphere_to_latlon(h_err_d, weights)

print(f"  Post-regrid v_ll_Linf:  LEGACY {np.abs(vn_p_ll).max():.3e},"
      f" DUOGRID {np.abs(vn_d_ll).max():.3e}")


def _plot_latlon(field, title, fname, cmap='RdBu_r', sym=True):
    """Plot lat-lon field as one panel."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if sym:
        vmax = np.abs(field).max()
        vmin = -vmax
    else:
        vmin, vmax = field.min(), field.max()
    im = ax.imshow(field, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-180, 180, -90, 90), aspect='auto')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    # Overlay lat/lon reference lines.
    for lat in (-60, -30, 0, 30, 60):
        ax.axhline(lat, color='gray', linewidth=0.3, alpha=0.5)
    for lon in (-90, 0, 90):
        ax.axvline(lon, color='gray', linewidth=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8, format='%.3f')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


_plot_latlon(vn_p_ll, f'v_north lat-lon: Production LEGACY C{N} day 1',
              'w2_vnorth_production_latlon.png')
_plot_latlon(vn_d_ll, f'v_north lat-lon: DUOGRID RK3 C{N} day 1',
              'w2_vnorth_duogrid_latlon.png')
_plot_latlon(h_err_p_ll, f'h_err lat-lon: Production LEGACY C{N} day 1',
              'w2_herr_production_latlon.png')
_plot_latlon(h_err_d_ll, f'h_err lat-lon: DUOGRID RK3 C{N} day 1',
              'w2_herr_duogrid_latlon.png')

print(f"\nImprovement: h_err max reduced by {(1-np.abs(h_err_d).max()/np.abs(h_err_p).max())*100:.0f}%")
print(f"  Production edge/int ratio: {vn_p[:,[0,-1],:].std() / (vn_p[:,2:-2,2:-2].std()+1e-30):.2f}")
print(f"  csw+dg edge/int ratio: {vn_d[:,[0,-1],:].std() / (vn_d[:,2:-2,2:-2].std()+1e-30):.2f}")
