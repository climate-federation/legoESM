"""Iter-828 diagnostic: W5 DUOGRID (post iter-808 sign-flip sync)
lat-lon visual.

Companion to `scripts/diag_iter824_w5_visual.py` (which produces
the LEGACY plots).  This script produces the DUOGRID equivalents
by setting `use_duogrid=True` while sharing the same cdgrid
config and IC construction.

iter-828b note (Codex review): the previous version of
diag_iter824_w5_visual.py had been edited to toggle USE_DUOGRID=
True in-place, which silently broke its "reproduce legacy W5
visuals" contract.  This separate script keeps the DUOGRID run
cleanly isolated.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test5)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
days = 3.0
dt = 300.0
n_steps = int(round(days * 86400 / dt))

grid = create_cubed_sphere(n=n, use_duogrid=True)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, div_damp=8.0 * _div_damp_cube(n),
    boundary_fix=True, damp_v=0.06, nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

sw = williamson_test5(grid)
u0 = 20.0
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(
    h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
model.set_initial_mass(state)

h0 = state.h
for _ in range(n_steps):
    state = model.step(state, dt)

h_np = np.asarray(state.h)
h_s_np = np.asarray(state.h_s)

from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
               + np.asarray(state.u_d)[:, :, 1:])
v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
               + np.asarray(state.v_d)[:, 1:, :])
v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc

weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
h_ll = apply_cubedsphere_to_latlon(h_np, weights)
vn_ll = apply_cubedsphere_to_latlon(v_north, weights)
h_change_ll = apply_cubedsphere_to_latlon(h_np - np.asarray(h0), weights)

print(f"Iter-828 W5 DUOGRID C{n} day {days:.0f} visual")
print(f"  h range: [{h_np.min():.1f}, {h_np.max():.1f}]")
print(f"  v_north Linf: {float(np.max(np.abs(vn_ll))):.3f}")
print(f"  |h - h0| RMS: {float(np.sqrt(np.mean((h_np-np.asarray(h0))**2))):.2f}")

OUT = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'fv3_visual')


def _plot(field, title, fname, cmap='RdBu_r', sym=True, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    if sym:
        m = np.abs(field).max()
        vmin, vmax = -m, m
    im = ax.imshow(field, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=(-180, 180, -90, 90), aspect='auto')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    for lat in (-60, -30, 0, 30, 60):
        ax.axhline(lat, color='gray', linewidth=0.3, alpha=0.5)
    for lon in (-90, 0, 90):
        ax.axvline(lon, color='gray', linewidth=0.3, alpha=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


_plot(h_ll, f'h final: W5 C{n} day {days:.0f} DUOGRID',
       'w5_h_duogrid_latlon.png', cmap='viridis', sym=False,
       vmin=h_ll.min(), vmax=h_ll.max())
_plot(h_change_ll, f'h(t=3d) - h(t=0): W5 C{n} DUOGRID',
       'w5_h_change_duogrid_latlon.png', cmap='RdBu_r', sym=True)
_plot(vn_ll, f'v_north: W5 C{n} day {days:.0f} DUOGRID',
       'w5_vnorth_duogrid_latlon.png', cmap='RdBu_r', sym=True)
