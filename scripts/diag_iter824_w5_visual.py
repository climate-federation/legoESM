"""Iter-824 diagnostic: W5 (mountain) lat-lon visual inspection.

Per Ralph loop step 5 (visual inspection on all 3 tests), iter-
824 completes the set by generating W5 h and v_north lat-lon
plots for C36 at day 3 LEGACY (matches iter-809's config).

W5 has no analytical time-dependent solution (mountain-driven
flow evolves non-trivially), so we plot h_final and v_north_final.
Any cube-sphere structural artifacts (stripes, polar bands,
cube-vertex features) would be visible against the mountain-wave
background.
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

# Iter-828b (Codex-review fix): restore iter-824's original LEGACY-
# only behaviour.  iter-828 had flipped USE_DUOGRID to True in
# this same script, breaking the "reproduce legacy W5 visuals"
# contract.  DUOGRID W5 visuals are now in
# `scripts/diag_iter828_w5_duogrid_visual.py` (separate script).
USE_DUOGRID = False

grid = create_cubed_sphere(n=n, use_duogrid=USE_DUOGRID)
# Iter-923: align with the iter-893 production matrix (adds
# `apply_fortran_xppm_boundary=True` to the iter-820 baseline).
# iter-921's audit showed W2 has a v_ll vs h_err Pareto trade-off
# under this swap; iter-923's W5 A/B at C36 day-3 measured all 7
# diagnostics within ±1 % — W5 is structurally insensitive to the
# iter-893 PPM boundary toggle.  Update keeps W5 visuals aligned
# with the live production sentinel.
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, div_damp=8.0 * _div_damp_cube(n),
    boundary_fix=True, damp_v=0.06, nord_v=2,
    apply_fortran_xppm_boundary=True)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

sw = williamson_test5(grid)
u0 = 20.0  # W5 uses u0=20 m/s (not 40 like W2)
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
hs_ll = apply_cubedsphere_to_latlon(h_s_np, weights)
vn_ll = apply_cubedsphere_to_latlon(v_north, weights)
h_change_ll = apply_cubedsphere_to_latlon(h_np - np.asarray(h0), weights)

print(f"Iter-824 W5 C{n} day {days:.0f} LEGACY visual inspection")
print(f"  h range: [{h_np.min():.1f}, {h_np.max():.1f}]")
print(f"  h_s range: [{h_s_np.min():.1f}, {h_s_np.max():.1f}]")
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


# Iter-828b (Codex review fix): restore the ORIGINAL iter-824
# filenames to preserve the legacy W5 contract.  The intermediate
# iter-828 edit had renamed `w5_h_change_latlon.png` to
# `w5_h_change_production_latlon.png`, breaking the legacy
# contract.  The three original filenames (one with and two
# without the "production" suffix) are preserved here.
_plot(h_ll, f'h final: W5 C{n} day {days:.0f} LEGACY',
       'w5_h_production_latlon.png', cmap='viridis', sym=False,
       vmin=h_ll.min(), vmax=h_ll.max())
_plot(h_change_ll, f'h(t=3d) - h(t=0): W5 C{n} LEGACY',
       'w5_h_change_latlon.png', cmap='RdBu_r', sym=True)
_plot(vn_ll, f'v_north: W5 C{n} day {days:.0f} LEGACY',
       'w5_vnorth_production_latlon.png', cmap='RdBu_r', sym=True)

print()
print("Interpretation cues (observational only):")
print("- h final should show mountain-driven wave structure.")
print("- h_change shows mountain-induced deviation from initial.")
print("- v_north Linf ~25 m/s is the mountain-wave amplitude, not")
print("  a cube-sphere artifact.")
print("- Check for stripes / corner blobs / polar bands superposed")
print("  on the wave pattern.")
