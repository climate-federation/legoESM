"""Iter-833 diagnostic: cosine bell error growth over multi-day
horizons at C36.

iter-776/781 characterized the 1-day residual.  iter-833 extends
the integration to 3 and 6 days to check:
- Linear vs nonlinear growth rate.
- Whether the bell's trajectory eventually crosses a cube vertex
  (iter-781 showed 1-day stays > 25° from nearest vertex).
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
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist_to_nearest_cube_vertex_deg(lat_deg, lon_deg):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    min_d = np.inf
    for vlat in CUBE_VERTEX_LATS_R:
        for vlon in CUBE_VERTEX_LONS_R:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return min_d


n = 36
dt = 1440.0
beta = jnp.pi / 4.0

grid = create_cubed_sphere(n)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, div_damp=_div_damp_cube(n),
    boundary_fix=True, damp_v=0.06, nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid
state = cosine_bell_cubesphere(grid, cdgrid, beta)
_, _, _, _, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
mass_init = float(jnp.sum(state.h * grid.area))

print(f"Iter-833 cosine bell C{n} multi-day Linf / L2 trajectory")
print(f"dt={dt}s, β=π/4, samples every 1 day")
print()
print(f"{'day':>4}  {'Linf':>10}  {'L2':>10}  {'h_max':>10}  {'bell GC-to-vertex':>18}")
print("-" * 62)

h = state.h
area_np = np.asarray(grid.area)

for day in range(0, 13):
    if day > 0:
        n_steps_per_day = int(round(86400 / dt))
        for _ in range(n_steps_per_day):
            h = transport_step(h, ut, vt, dt, cdgrid, mass_target=mass_init)
    t_s = day * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    h_np = np.asarray(h)
    h_ex_np = np.asarray(h_exact)
    err = h_np - h_ex_np
    Linf = float(np.max(np.abs(err)))
    L2_num = float(np.sum(area_np * err ** 2))
    L2_den = float(np.sum(area_np * h_ex_np ** 2))
    L2 = float(np.sqrt(L2_num / L2_den)) if L2_den > 0 else np.nan
    h_max = float(np.max(h_np))
    # Bell location (argmax of exact).
    idx_ex = np.unravel_index(np.argmax(h_ex_np), h_ex_np.shape)
    bell_lat = float(np.rad2deg(np.asarray(grid.lat)[idx_ex]))
    bell_lon = float(np.rad2deg(np.asarray(grid.lon)[idx_ex]))
    if bell_lon > 180:
        bell_lon -= 360
    bell_gc = _gc_dist_to_nearest_cube_vertex_deg(bell_lat, bell_lon)
    print(f"  {day:>2}  {Linf:>10.3e}  {L2:>10.3e}  {h_max:>10.2f}  {bell_gc:>17.2f}°")

print()
print("Interpretation cues (observational only):")
print("- Linf growth pattern: linear in day = dispersion at constant")
print("  rate.  Sharp jumps at specific days = cube-vertex crossings.")
print("- bell GC-to-vertex: when < 5°, bell is near a cube vertex.")
print("- h_max < 1000 = amplitude diffusion.")
