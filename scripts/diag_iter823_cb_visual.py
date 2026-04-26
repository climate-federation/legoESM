"""Iter-823 diagnostic: cosine bell visual inspection at C36 1-day.

Per Ralph loop step 5 ("Perform careful visual inspection.  There
must be no panel-edge, corner, seam, halo, striping, or ringing
artifacts"), iter-823 generates h, h_err, and exact h lat-lon
plots for the cosine bell 1-day C36 run.

Shows whether the ~121 m Linf residual is:
- Smoothly distributed (expected structural truncation)
- Concentrated at cube-vertex stripes (like W2)
- Has other unexpected patterns
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.fv3_sw_core import _d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 1440.0
days = 1.0
beta = jnp.pi / 4.0
n_steps = int(round(days * 86400 / dt))

grid = create_cubed_sphere(n)
# Iter-924: align with iter-893 production matrix
# (`apply_fortran_xppm_boundary=True`).  iter-921→923 audit showed
# W2 has a v_ll vs h_err Pareto trade-off under this swap; iter-924's
# cosine bell A/B at C36 day-1 measured all 4 diagnostics within
# ±0.13 % — cosine bell pure-transport is structurally insensitive
# to the iter-893 PPM boundary toggle.
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, div_damp=_div_damp_cube(n),
    boundary_fix=True, damp_v=0.06, nord_v=2,
    apply_fortran_xppm_boundary=True)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid
state = cosine_bell_cubesphere(grid, cdgrid, beta)
_, _, _, _, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
mass_init = float(jnp.sum(state.h * grid.area))
h = state.h
for _ in range(n_steps):
    h = transport_step(h, ut, vt, dt, cdgrid, mass_target=mass_init,
                       apply_fortran_xppm_boundary=True)

t_s = days * 86400.0
h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
h_num = np.asarray(h)
h_ex = np.asarray(h_exact)
h_err = h_num - h_ex

print(f"Iter-823 cosine bell C{n} 1-day visual inspection (LEGACY)")
print(f"  max |h_num|  = {float(np.max(np.abs(h_num))):.2f}")
print(f"  max h_exact  = {float(np.max(h_ex)):.2f}")
print(f"  Linf h_err   = {float(np.max(np.abs(h_err))):.2f}")
print(f"  L2 (rel)     = {float(np.sqrt(np.mean(h_err**2)) / np.sqrt(np.mean(h_ex**2))):.3e}")

# Regrid to lat-lon for clean visualization.
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
h_num_ll = apply_cubedsphere_to_latlon(h_num, weights)
h_ex_ll = apply_cubedsphere_to_latlon(h_ex, weights)
h_err_ll = apply_cubedsphere_to_latlon(h_err, weights)

OUT = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'fv3_visual')


def _plot(field, title, fname, cmap='viridis', sym=False, vmin=None, vmax=None):
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


_plot(h_num_ll, f'h numerical: cosine bell C{n} day 1',
       'cb_h_production_latlon.png', cmap='viridis',
       vmin=0, vmax=1000)
_plot(h_ex_ll, f'h exact: cosine bell C{n} day 1',
       'cb_h_exact_latlon.png', cmap='viridis',
       vmin=0, vmax=1000)
_plot(h_err_ll, f'h error (num - exact): cosine bell C{n} day 1',
       'cb_h_err_latlon.png', cmap='RdBu_r', sym=True)

print()
print("Interpretation cues (observational only):")
print("- h_num and h_exact should match visually.")
print("- h_err should be localised at the bell's leading/trailing edges")
print("  if dominated by dispersion/dissipation, or structured at cube")
print("  vertices if dominated by iter-796-style cancellation residual.")
