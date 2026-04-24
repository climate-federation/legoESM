"""Iter-782 diagnostic: `_d2a2c_vect` fidelity for the solid-body
rotation initial condition at C36.

Iter-781 extended iter-780's end-of-day snapshot into an 11-sample
trajectory and confirmed peak |err| grows monotonically in the
cosine bell C36 1-day run.  One outstanding candidate mechanism
(flagged unaudited since iter-776b) is whether `_d2a2c_vect`
itself introduces nontrivial error at cube vertices BEFORE any
time integration begins.

`_d2a2c_vect` is called ONCE at t=0 for the cosine bell: it takes
the analytical solid-body rotation D-grid (u_d, v_d) and returns
C-grid contravariant (ut, vt) which are then held constant for
the whole 1-day transport.  Any cube-vertex inaccuracy in
`_d2a2c_vect` therefore imprints a CONSTANT bias on the transport
winds that persists throughout the run.

This diagnostic computes:
  - ut, vt from `_d2a2c_vect` applied to the solid-body rotation IC
  - ut_exact, vt_exact analytically at the same C-grid stagger
    positions using the grid's covariant→contravariant formulas
  - |ut - ut_exact| and |vt - vt_exact| maps
  - the peak-error cell's GC-distance to the nearest cube vertex

If the peak |err| in (ut, vt) lives at or very near a cube vertex
(GC ≲ 2 cell widths), candidate "d2a2c cube-vertex gap" is
directly relevant.  If the peak is mid-face (GC ≳ 5 cell widths),
that candidate is a poor fit for the observed cosine bell error.

Scope: observational only, no source-code change, no new sentinel.
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
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, _rotation_winds_geo)


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


def _cell_dist_to_face_edge(ci, cj, n):
    return min(ci, n - 1 - ci, cj, n - 1 - cj)


n = 36
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

# ----- Python _d2a2c_vect on the solid-body rotation IC -----
state = cosine_bell_cubesphere(grid, cdgrid, beta)
ua, va, uc, vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
ut = np.asarray(ut)
vt = np.asarray(vt)

# ----- Analytical ut_exact / vt_exact at C-grid stagger -----
# ut lives at y-edge (i+1/2, j), same stagger as v_d.  Compute the
# covariant components from the analytical geographic (u_east,
# v_north) at that stagger, then apply the same covariant→
# contravariant formula `_d2a2c_vect` uses at line 400:
#   ut = (uc - v_d * cosa_u) * rsin_u
# Here we substitute the analytical covariant u at y-edge for uc
# and the analytical covariant v at y-edge for v_d.
u_e_y, v_n_y = _rotation_winds_geo(
    cdgrid.lon_edge_y, cdgrid.lat_edge_y, grid.radius, beta)
u_cov_y = (cdgrid.cos_angle_edge_y * u_e_y
           + cdgrid.sin_angle_edge_y * v_n_y)
v_cov_y = (-cdgrid.sin_angle_edge_y * u_e_y
           + cdgrid.cos_angle_edge_y * v_n_y)
ut_exact = (u_cov_y - v_cov_y * cdgrid.cosa_u) * cdgrid.rsin_u
ut_exact = np.asarray(ut_exact)

# vt lives at x-edge (i, j+1/2), same stagger as u_d.  Analogous
# construction with cosa_v, rsin_v.
u_e_x, v_n_x = _rotation_winds_geo(
    cdgrid.lon_edge_x, cdgrid.lat_edge_x, grid.radius, beta)
u_cov_x = (cdgrid.cos_angle_edge_x * u_e_x
           + cdgrid.sin_angle_edge_x * v_n_x)
v_cov_x = (-cdgrid.sin_angle_edge_x * u_e_x
           + cdgrid.cos_angle_edge_x * v_n_x)
vt_exact = (v_cov_x - u_cov_x * cdgrid.cosa_v) * cdgrid.rsin_v
vt_exact = np.asarray(vt_exact)

# ----- Locate peak errors -----
ut_err = np.abs(ut - ut_exact)
vt_err = np.abs(vt - vt_exact)

lat_y = np.rad2deg(np.asarray(cdgrid.lat_edge_y))
lon_y = np.rad2deg(np.asarray(cdgrid.lon_edge_y))
lon_y = np.where(lon_y > 180.0, lon_y - 360.0, lon_y)
lat_x = np.rad2deg(np.asarray(cdgrid.lat_edge_x))
lon_x = np.rad2deg(np.asarray(cdgrid.lon_edge_x))
lon_x = np.where(lon_x > 180.0, lon_x - 360.0, lon_x)


def _report_peak(label, err, lat, lon, stagger_shape):
    idx = np.unravel_index(np.argmax(err), err.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    peak = float(err[face, ci, cj])
    latd = float(lat[face, ci, cj])
    lond = float(lon[face, ci, cj])
    gc = _gc_dist_to_nearest_cube_vertex_deg(latd, lond)
    # Stagger_shape is (n_i, n_j).  "cell-to-face-edge" distance in
    # stagger indices.
    ced = min(ci, stagger_shape[0] - 1 - ci,
              cj, stagger_shape[1] - 1 - cj)
    print(f"  {label}  peak={peak:.3e}  face={face}  (i,j)=({ci},{cj})  "
          f"lat={latd:+.2f}°  lon={lond:+.2f}°  "
          f"GC-to-vertex={gc:.2f}°  cell-to-edge={ced}")


print(f"Iter-782 `_d2a2c_vect` fidelity at C36, β=π/4 solid-body rotation")
print(f"ut shape={ut.shape}, vt shape={vt.shape}")
print()
print(f"Peak |ut - ut_exact| and |vt - vt_exact|:")
_report_peak("ut", ut_err, lat_y, lon_y, ut.shape[1:])
_report_peak("vt", vt_err, lat_x, lon_x, vt.shape[1:])
print()

# Summary stats: cube-vertex-proximity distribution
ut_thr = 0.5 * float(np.max(ut_err))
vt_thr = 0.5 * float(np.max(vt_err))

def _frac_near_vertices(err, lat, lon, thr, max_deg):
    # Fraction of "hot" points (err > threshold) within max_deg GC of
    # any cube vertex.
    hot_mask = err > thr
    hot_count = int(np.sum(hot_mask))
    if hot_count == 0:
        return 0, 0
    latd = lat[hot_mask]
    lond = lon[hot_mask]
    near = 0
    for i in range(hot_count):
        gc = _gc_dist_to_nearest_cube_vertex_deg(float(latd[i]),
                                                  float(lond[i]))
        if gc < max_deg:
            near += 1
    return hot_count, near


for max_deg in (5.0, 10.0, 20.0):
    ut_hot, ut_near = _frac_near_vertices(ut_err, lat_y, lon_y,
                                           ut_thr, max_deg)
    vt_hot, vt_near = _frac_near_vertices(vt_err, lat_x, lon_x,
                                           vt_thr, max_deg)
    print(f"Within {max_deg:>4.1f}° of any cube vertex "
          f"(hot := |err| > 0.5 * peak):")
    print(f"  ut: {ut_near:>4d} / {ut_hot:>5d} hot points "
          f"({100.0 * ut_near / max(ut_hot, 1):.1f}%)")
    print(f"  vt: {vt_near:>4d} / {vt_hot:>5d} hot points "
          f"({100.0 * vt_near / max(vt_hot, 1):.1f}%)")

print()
print(f"Absolute scale (for perspective):")
print(f"  max |ut_exact| = {float(np.max(np.abs(ut_exact))):.3e} m/s")
print(f"  max |vt_exact| = {float(np.max(np.abs(vt_exact))):.3e} m/s")
print(f"  max |ut - ut_exact| = {float(np.max(ut_err)):.3e} m/s "
      f"({100.0 * float(np.max(ut_err)) / float(np.max(np.abs(ut_exact))):.2f}% relative)")
print(f"  max |vt - vt_exact| = {float(np.max(vt_err)):.3e} m/s "
      f"({100.0 * float(np.max(vt_err)) / float(np.max(np.abs(vt_exact))):.2f}% relative)")

print()
print("What iter-782 DOES measure (observational only):")
print("- Absolute |err| in ut/vt after one `_d2a2c_vect` call on")
print("  the analytical solid-body rotation IC at t=0.")
print("- Peak-error location and its GC-distance to nearest cube")
print("  vertex.")
print("- Hot-point (|err| > 0.5*peak) clustering near cube vertices.")
print("What it does NOT establish:")
print("- Whether the measured `_d2a2c_vect` error is large enough to")
print("  explain the cosine bell Linf ~ 121 at t=1 day (needs")
print("  advection of h-field through an error-biased wind field).")
print("- Whether the measured error comes from `fill_corner_region`")
print("  (corner Lagrange) or from the 4th-order A→C stencil.")
print("- Whether other operators (`transport_step`, `fv_tp_2d`)")
print("  contribute additional error on top of this.")
