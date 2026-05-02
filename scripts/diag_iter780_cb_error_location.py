"""Iter-780 diagnostic: cosine bell error-location sweep at
C16, C24, C36, C48 (fixed dt=1350s).

Iter-779 established that the cosine bell error plateau above
C24 is NOT an artifact of dt scaling.  One of iter-778's
remaining candidate mechanisms is:
  (c) cube-vertex-localized error set by face count (constant 8)
      whose shape shrinks but whose integrated magnitude does not.

Iter-780 measures WHERE the peak |h - h_exact| lives at each
resolution:
- face index
- face-local (i, j)
- great-circle distance to the nearest of the 8 cube vertices
  (at (±arcsin(1/sqrt(3)) ≈ ±35.26°, ±45°/±135°))
- cell distance (in grid-cells) to the nearest face edge/corner

If the peak-error GC-distance to a cube vertex stays approximately
constant across resolutions (same few degrees regardless of n),
candidate (c) is supported: the error is locked to the cube-
geometry, not to the grid discretization.

If the peak-error position drifts randomly with n, candidate (c)
is not supported.
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

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.fv3_sw_core import _d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist_to_nearest_cube_vertex_deg(lat_deg, lon_deg):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    min_d = np.inf
    for vlat in CUBE_VERTEX_LATS:
        for vlon in CUBE_VERTEX_LONS:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return min_d


def run_and_locate(n, dt=1350.0, days=1):
    n_steps = int(round(days * 86400 / dt))
    dt_exact = days * 86400 / n_steps
    beta = jnp.pi / 4.0
    grid = create_cubed_sphere(n)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=_div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid
    state = cosine_bell_cubesphere(grid, cdgrid, beta)
    _ua, _va, _uc, _vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
    mass_init = float(jnp.sum(state.h * grid.area))
    h = state.h
    for _ in range(n_steps):
        h = transport_step(h, ut, vt, dt_exact, cdgrid,
                            mass_target=mass_init)
    t_s = days * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    err = np.asarray(h) - np.asarray(h_exact)
    abs_err = np.abs(err)
    idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    lat = float(np.rad2deg(np.asarray(grid.lat)[face, ci, cj]))
    lon = float(np.rad2deg(np.asarray(grid.lon)[face, ci, cj]))
    if lon > 180: lon -= 360
    gc = _gc_dist_to_nearest_cube_vertex_deg(lat, lon)
    # Cell distance to nearest face edge.
    min_i = min(ci, n - 1 - ci)
    min_j = min(cj, n - 1 - cj)
    cell_dist_to_edge = min(min_i, min_j)
    peak_err = float(abs_err[face, ci, cj])
    return (face, ci, cj, lat, lon, gc, cell_dist_to_edge, peak_err)


print(f"Iter-780 cosine bell peak-error location at C16/C24/C36/C48")
print(f"β=π/4, 1 day, dt=1350s")
print()
print(f"{'n':>3}  {'face':>4}  {'(i, j)':>10}  {'lat':>7}  {'lon':>8}  "
      f"{'GC to vertex':>14}  {'cell to edge':>12}  {'peak|err|':>10}")
print("-" * 82)
results = []
for n in (16, 24, 36, 48):
    face, ci, cj, lat, lon, gc, ced, pe = run_and_locate(n)
    results.append((n, face, ci, cj, lat, lon, gc, ced, pe))
    print(f"  {n:>2}  {face:>3}  ({ci:>2},{cj:>2})  "
          f"{lat:>6.2f}°  {lon:>7.2f}°  {gc:>12.2f}°  "
          f"{ced:>11}  {pe:>10.3e}")

# Observational summary — numerical reportage only; no mechanism
# attribution.  Observation labels must remain neutral.
print()
print("Measured summary across the 4 resolutions:")
gc_values = [r[6] for r in results]
ced_values = [r[7] for r in results]
pe_values = [r[8] for r in results]
print(f"  GC-distance to nearest cube vertex : "
      f"min={min(gc_values):.2f}°, max={max(gc_values):.2f}°, "
      f"span={max(gc_values) - min(gc_values):.2f}°")
print(f"  cell-distance to face edge         : "
      f"min={min(ced_values)}, max={max(ced_values)}")
print(f"  peak |err|                          : "
      f"min={min(pe_values):.3e}, max={max(pe_values):.3e}")
print()
print("What this report DOES show (observational only):")
print("- At each n, the peak |h-h_exact| position (face, i, j) is")
print("  reported.")
print("- GC distance to the nearest of 8 cube vertices is reported.")
print("What it does NOT establish:")
print("- Whether the peak-error cell is also where the error was")
print("  ACCUMULATED (vs a downstream collection of upstream errors).")
print("- Whether the peak-error position is stable in time (this is")
print("  a single end-of-day snapshot, not a trajectory).")
print("- Which of iter-778's candidate mechanisms (a) / (c) / (d) is")
print("  supported — distinguishing requires further diagnostics that")
print("  iter-780 does not run.")
