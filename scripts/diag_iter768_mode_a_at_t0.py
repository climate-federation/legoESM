"""Iter-768 diagnostic: two-point `v_ll_Linf` measurement at t=0
and t=1 day on W2 C36 with the canonical matrix config.  Reports
observables only; does not decompose them into causal components.

**Scope clarification (iter-768b, Codex stop-time review).**  The
original iter-768 framing ("IC 5 %, dynamics 95 %") treated the
ratio of two `v_ll_Linf` observables as a causal split — it is
not.  This docstring and the runtime output have been rewritten
to report only observables.

What this diagnostic measures:
- `v_ll_Linf` at t=0, constructed from the analytic W2 IC.  Also
  reports `u_d err Linf` and `v_d err Linf` vs the analytic IC
  (both are exactly 0 — the stored state matches the analytic
  reference).
- `v_ll_Linf` at t=1 day after 288 RK3 steps with the canonical
  matrix config.
- The ratio (t=1 day) / (t=0) as a single scalar.

What this diagnostic does NOT establish:
- Which of the dycore, the measurement pipeline (D-grid →
  cell-centre-avg → rotate → regrid), the IC construction, or
  couplings among them is responsible for the 1-day observable.
  A single-measurement ratio of observables cannot separate
  those.  Iter-769+ ablation work is needed (e.g. re-run with an
  alternate regrid only, re-run with an alternate rotation only,
  etc., and compare 1-day `v_ll_Linf` values).

Uses the canonical matrix measurement path identical to
`scripts/run_atmosphere_test_matrix.py` (IC, dt, norms, v_north
convention), same as iter-766c / iter-767 diagnostics.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import cell_centre_angles_from_4edge
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2, williamson_test2_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 300.0
days = 1
n_steps = int(days * 86400 / dt)

grid = create_cubed_sphere(n)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

sw = williamson_test2(grid)
exact = williamson_test2_exact(grid, days * 86400.0)
weights = get_cubedsphere_to_latlon_weights(n, 360, 181)

# Canonical matrix config.
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=8.0 * _div_damp_cube(n),
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2,
)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

# IC: same analytic edge-midpoint D-grid winds the matrix uses.
u_d_ic = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d_ic = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(
    h=sw.h.data, u_d=u_d_ic, v_d=v_d_ic, h_s=sw.h_s.data)
model.set_initial_mass(state)


def measure(state, label: str):
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    v_north_linf = float(np.max(np.abs(v_north)))
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))

    # Also raw D-grid: the D-grid per-face winds themselves (no
    # cell-centre averaging, no regrid).  Reconstruct v_north at the
    # D-grid corners using cdgrid corner angles.
    u_d = np.asarray(state.u_d)     # (6, n, n+1)
    v_d = np.asarray(state.v_d)     # (6, n+1, n)
    # Can't directly rotate u_d+v_d because they're at different
    # stagger points.  Instead, use the face-local grid-aligned
    # representation directly and measure its deviation from the
    # exact W2 values.
    u_d_exact = np.asarray(u_d_ic)
    v_d_exact = np.asarray(v_d_ic)
    u_d_err_linf = float(np.max(np.abs(u_d - u_d_exact)))
    v_d_err_linf = float(np.max(np.abs(v_d - v_d_exact)))

    print(f"[{label}]")
    print(f"  v_north_Linf (per-face, cell-centre) = {v_north_linf:.4e}")
    print(f"  v_ll_Linf    (post-regrid lat-lon)    = {v_ll_linf:.4e}")
    print(f"  u_d  err Linf (vs analytic IC)        = {u_d_err_linf:.4e}")
    print(f"  v_d  err Linf (vs analytic IC)        = {v_d_err_linf:.4e}")
    return v_ll_linf


# (1) Measure v_ll_Linf at t=0 (IC only).
v_ll_0 = measure(state, "t=0 (IC only, no time steps)")
print()

# (2) Integrate 1 day and measure.
for _ in range(n_steps):
    state = model.step(state, dt)
v_ll_day = measure(state, "t=1 day (after 288 steps)")
print()

# Summary.
ratio = v_ll_day / max(v_ll_0, 1e-12)
print(f"Two-point v_ll_Linf observables on W2 C36 1d, canonical")
print(f"matrix config (NO causal attribution claimed):")
print(f"  t=0      v_ll_Linf  = {v_ll_0:.4e} m/s")
print(f"    (u_d/v_d err Linf = 0 — stored state matches analytic IC)")
print(f"  t=1 day  v_ll_Linf  = {v_ll_day:.4e} m/s")
print(f"  ratio (t=1 day / t=0) = {ratio:.2f}x")
print()
print("This is a two-point observable measurement.  The ratio does")
print("not by itself identify which of the dycore, the measurement")
print("pipeline, the IC construction, or couplings among them is")
print("responsible for the t=1-day observable.  Mechanism")
print("separation requires iter-769+ ablations.")

# Peak locator: print top 5 |v_ll| values + their (lat, lon).
# Useful for future iters to verify a fix REDUCED the peak amplitude
# rather than DISPLACED it to a different location.  iter-762
# localized mode A to the 8 cube vertices lat=±35.26°, lon=±45°/±135°;
# these should be the top 5+ peaks.
ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
               + np.asarray(state.u_d)[:, :, 1:])
v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
               + np.asarray(state.v_d)[:, 1:, :])
v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
v_ll = np.asarray(apply_cubedsphere_to_latlon(v_north, weights))
# v_ll has shape (n_lat=181, n_lon=360).
lon_grid = np.linspace(-180, 180, 360, endpoint=False)
lat_grid = np.linspace(-90, 90, 181)
flat_idx = np.argsort(-np.abs(v_ll.ravel()))[:10]
print()
print("Top 10 |v_ll| locations at t=1 day:")
print(f"  {'|v_ll|':>10}  {'lat':>7}  {'lon':>7}  "
      f"{'GC_dist_to_vertex_deg':>22}  {'near_cube_vertex?':>18}")
# 8 standard gnomonic cube-sphere vertices on the unit sphere at
# (±1/sqrt(3), ±1/sqrt(3), ±1/sqrt(3)) normalized — lat = arcsin(1/sqrt(3))
# ≈ 35.2644°, lon in {±45°, ±135°}.
cube_vertex_lat_rad = np.arcsin(1.0 / np.sqrt(3.0))
cube_vertex_lats = np.array([cube_vertex_lat_rad, -cube_vertex_lat_rad])
cube_vertex_lons = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))

def _gc_dist_deg(lat_deg, lon_deg):
    """Great-circle distance from (lat_deg, lon_deg) to the NEAREST
    of the 8 cube vertices, returned in degrees.  Avoids the
    anisotropic degree-box (a 5° box is ~555 km in lat but ~450 km
    in lon at lat=35°)."""
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    min_d = np.inf
    for vlat in cube_vertex_lats:
        for vlon in cube_vertex_lons:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            # Haversine
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return min_d

for idx in flat_idx:
    i_lat, i_lon = np.unravel_index(idx, v_ll.shape)
    lat = lat_grid[i_lat]
    lon = lon_grid[i_lon]
    gc = _gc_dist_deg(lat, lon)
    # 5° great-circle (~555 km) threshold: accounts for (a) 1° regrid
    # binning, (b) cell-centre offset from the cube-vertex corner
    # (~1-2 cells at C36 ≈ ~2.5°-5° spacing).  The corner cell
    # indices (0, 0)/(0, n-1)/etc. have centres that many degrees
    # inside the face, not AT the cube-vertex point.
    near_cv = "YES" if gc < 5.0 else "no"
    print(f"  {np.abs(v_ll[i_lat, i_lon]):>10.4e}  {lat:>6.1f}°  "
          f"{lon:>6.1f}°  {gc:>20.2f}°  {near_cv:>18}")
