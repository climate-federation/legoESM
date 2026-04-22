"""Iter-770 diagnostic: face-local (face, i, j) index identification
for the top-10 |v_north| peaks on the per-face cell-centre field.

Iter-768's peak locator operates on the REGRIDDED `v_ll` field
which has shape (n_lat=181, n_lon=360).  That confirms the peaks
are within ~3.3 degrees great-circle of the 8 cube vertices.  But
the lat/lon positions cannot tell which FACE-LOCAL INDEX the
artifact lives at.

This diagnostic operates on the PRE-REGRID `v_north` field of
shape (6, n, n) and reports, for each top-10 |v_north| peak:
- face index (0..5)
- face-local (i, j) with 0-based row/column indexing
- chebyshev distance to the nearest cube-corner cell position
  {(0,0), (0,n-1), (n-1,0), (n-1,n-1)}
- whether the cell is AT a cube corner, on an edge row/col, or
  strictly interior

The zone classification per cell is:
- CORNER: cell is at one of the 4 cube-corner positions
  (0,0) / (0,n-1) / (n-1,0) / (n-1,n-1).
- EDGE: cell is on a boundary row/col (index 0 or n-1) but is
  NOT a corner cell.
- NEAR-CORNER: cell is strictly interior (not on row 0/n-1 or
  col 0/n-1) with Chebyshev distance 1-2 from a corner cell.
- INTERIOR: all other cells.

The zone tally across the top 10 peaks is reported as counts
only; it does not claim any one zone is "dominant" or that
mode A is "concentrated" in a particular zone — those would be
interpretive claims beyond what a peak-location scan shows.

Uses the canonical matrix measurement path (same IC, dt, config,
v_north convention as iter-766c/767/768/769 diagnostics).  Does
NOT compute area-weighted L2/Linf norms — iter-770 reports only
the per-cell peak locations.
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
from legoesm.grids.cubed_sphere_cdgrid import cell_centre_angles_from_4edge
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 300.0
days = 1
n_steps = int(days * 86400 / dt)

grid = create_cubed_sphere(n)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

sw = williamson_test2(grid)

cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=8.0 * _div_damp_cube(n),
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2,
)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(
    h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
model.set_initial_mass(state)

for _ in range(n_steps):
    state = model.step(state, dt)

ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
               + np.asarray(state.u_d)[:, :, 1:])
v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
               + np.asarray(state.v_d)[:, 1:, :])
v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc  # (6, n, n)
lat = np.asarray(cdgrid.base.lat)  # (6, n, n)
lon = np.asarray(cdgrid.base.lon)  # (6, n, n)

flat_idx = np.argsort(-np.abs(v_north.ravel()))[:10]

corner_positions = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]


def _classify(i, j):
    """Return (zone, cheb_dist_to_corner)."""
    min_d = min(max(abs(i - ci), abs(j - cj))
                 for ci, cj in corner_positions)
    is_corner = (i, j) in corner_positions
    is_edge = (i in (0, n-1)) or (j in (0, n-1))
    if is_corner:
        zone = "CORNER"
    elif is_edge:
        zone = "EDGE"
    elif min_d <= 2:
        zone = "NEAR-CORNER"
    else:
        zone = "INTERIOR"
    return zone, min_d


print(f"Top 10 |v_north| peaks at t=1 day (pre-regrid, per-face frame):")
print()
print(f"{'rank':>4}  {'|v_north|':>10}  {'face':>4}  {'(i, j)':>8}  "
      f"{'lat':>8}  {'lon':>8}  {'zone':>12}  {'cheb_to_corner':>14}")
print("-" * 90)
for rank, idx in enumerate(flat_idx, start=1):
    face, i, j = np.unravel_index(idx, v_north.shape)
    zone, cheb = _classify(int(i), int(j))
    lat_deg = float(np.rad2deg(lat[face, i, j]))
    lon_deg = float(np.rad2deg(lon[face, i, j]))
    # Normalize lon to (-180, 180]
    if lon_deg > 180:
        lon_deg -= 360
    print(f"{rank:>4}  {abs(v_north[face, i, j]):>10.4e}  "
          f"{int(face):>4}  ({int(i):>2},{int(j):>2})  "
          f"{lat_deg:>7.2f}°  {lon_deg:>7.2f}°  "
          f"{zone:>12}  {int(cheb):>14}")

# Zone tally for the top 10 peaks (counts only; no interpretive
# claim about mechanism or "dominance").  Iter-770 reports zones
# observationally; the commit message / review-doc may list these
# as inputs to iter-771+ candidate selection without asserting
# causation here.
zones = []
for idx in flat_idx:
    face, i, j = np.unravel_index(idx, v_north.shape)
    zone, _ = _classify(int(i), int(j))
    zones.append(zone)

from collections import Counter
zone_counts = Counter(zones)
print()
print(f"Zone tally across top 10 peaks (counts only, no mechanism claim):")
for z, count in zone_counts.most_common():
    print(f"  {z}: {count}")
