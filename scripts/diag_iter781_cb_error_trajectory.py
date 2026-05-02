"""Iter-781 diagnostic: cosine bell peak-error TRAJECTORY at
C36, sampled every 0.1 day.

Iter-780 reported only the end-of-day (t=1.0) peak-error
location.  That single snapshot cannot distinguish between
error that accumulates UNIFORMLY with time and error that
ACCUMULATES at specific moments (e.g., when the bell crosses
a cube vertex).  Iter-781 samples every 0.1 day to build a
trajectory of peak-error position + GC-distance-to-nearest-
cube-vertex + peak |err|.

Hypothesis (iter-778 candidate (c)): "cube-vertex-localized
error set by face count (constant 8) whose shape shrinks but
whose integrated magnitude does not."  If (c) is the mechanism,
the trajectory should show peak |err| SPIKES at specific times
when the bell crosses a cube vertex (GC-to-vertex becomes small
transiently), while peak |err| grows modestly between crossings.

If peak |err| grows approximately uniformly throughout the
trajectory (no spikes at cube crossings), candidate (c) is less
likely and candidate (a) "resolution-invariant structural error"
or (d) "PPM limiter" become more likely.

Scope: 1 resolution (C36), 1 IC, 1 horizon, β=π/4.  Same as
iter-780's config.  Samples at t = 0.0, 0.1, 0.2, ..., 1.0
(11 snapshots including t=0 which is identity error).
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


n = 36
# iter-781b: choose dt so that sample_every (0.1 day = 8640s) is
# an integer multiple of dt.  Previous iter-781 had dt=1350s which
# gave 6 steps × 1350s = 8100s per "0.1 day" label — the
# reported times were 6.25% SHORTER than labeled.  Codex caught
# this as time-misalignment.  Fix: dt=1440s so that 6*1440=8640s
# exactly equals 0.1 day.
dt = 1440.0
days = 1.0
sample_every = 0.1  # days — now exactly 6 dt steps
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

# Also need the bell's position trajectory for reference.
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

print(f"Iter-781 cosine bell error trajectory at C36, dt={dt}s")
print(f"β=π/4, 1 day, sampled every {sample_every} days")
print()
print(f"{'t (d)':>7}  {'face':>4}  {'(i, j)':>10}  {'lat':>7}  {'lon':>8}  "
      f"{'GC-to-vertex':>13}  {'peak|err|':>11}  {'bell_pos_GC':>12}")
print("-" * 85)

h = state.h
# iter-781b: use EXACT integer step count and assert alignment.
_raw = sample_every * 86400 / dt
steps_per_sample = int(round(_raw))
assert abs(_raw - steps_per_sample) < 1e-9, (
    f"sample_every={sample_every} day and dt={dt} s do not "
    f"produce an integer steps_per_sample (got {_raw}).  "
    f"Adjust dt so sample_every*86400 is an integer multiple.")
total_samples = int(days / sample_every) + 1

for sample_idx in range(total_samples):
    t_s = sample_idx * sample_every * 86400
    t_d = t_s / 86400
    if sample_idx > 0:
        for _ in range(steps_per_sample):
            h = transport_step(h, ut, vt, dt, cdgrid,
                                mass_target=mass_init)
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    err = np.asarray(h) - np.asarray(h_exact)
    abs_err = np.abs(err)
    idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    lat = float(np.rad2deg(np.asarray(grid.lat)[face, ci, cj]))
    lon = float(np.rad2deg(np.asarray(grid.lon)[face, ci, cj]))
    if lon > 180: lon -= 360
    gc = _gc_dist_to_nearest_cube_vertex_deg(lat, lon)
    peak_err = float(abs_err[face, ci, cj])

    # Also report where the bell's CENTER is now, as GC-distance
    # to nearest cube vertex.
    # Bell centre is at reverse-rotation of (0°, 0°) by angle u0*t/R.
    angle = u0 * t_s / float(grid.radius)
    # Initial bell center: lat=0, lon=-90 deg (PL07 convention).
    # After time t with rotation angle=angle around axis beta=pi/4,
    # the centre moves.  Use a simple forward-rotation approx.
    bell_center_lat_deg = np.rad2deg(np.arcsin(
        np.sin(np.deg2rad(-90.0) * 0) * 0))  # Placeholder
    # Simplified: use the exact peak of h_exact as proxy for centre.
    h_ex_np = np.asarray(h_exact)
    idx_ex = np.unravel_index(np.argmax(h_ex_np), h_ex_np.shape)
    bell_lat = float(np.rad2deg(np.asarray(grid.lat)[idx_ex]))
    bell_lon = float(np.rad2deg(np.asarray(grid.lon)[idx_ex]))
    if bell_lon > 180: bell_lon -= 360
    bell_gc = _gc_dist_to_nearest_cube_vertex_deg(bell_lat, bell_lon)

    print(f"  {t_d:>5.2f}  {face:>3}  ({ci:>2},{cj:>2})  "
          f"{lat:>6.2f}°  {lon:>7.2f}°  {gc:>11.2f}°  "
          f"{peak_err:>11.3e}  {bell_gc:>10.2f}°")

print()
print("Interpretation cues (observational only):")
print("- Trends in 'GC-to-vertex' column show whether peak error")
print("  position tracks cube vertices over time.")
print("- Trends in 'peak|err|' column show growth rate: spikes at")
print("  cube crossings would support candidate (c); approximately")
print("  monotone growth suggests (a) or (d).")
print("- 'bell_pos_GC' = GC distance of the bell CENTRE (taken from")
print("  the exact solution's peak) to the nearest cube vertex.")
print("  When this is small, the bell is passing over a cube vertex.")
print("Iter-781 is observational only; the report makes no mechanism")
print("claim beyond the measured numbers.")
