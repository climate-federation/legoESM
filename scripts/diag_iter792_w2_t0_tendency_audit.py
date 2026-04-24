"""Iter-792 diagnostic: W2 t=0 tendency (du_dt, dv_dt) cube-vertex
audit.

Iter-791 ruled out `fv3_cc2c` as the direct W2 mode-A source (its
peak error on analytical input is at the POLE, not cube vertices).
Iter-792 tests the next suspect: the momentum tendency operator
`fv3_sw_tendencies` which combines KE gradient, pressure gradient,
vorticity flux at D-grid corners via `_arakawa_lamb_gradient`.

At t=0 for W2 solid-body rotation:
  - The ANALYTICAL tendencies are du_d/dt = 0, dv_d/dt = 0
    (steady-state geostrophic balance).
  - ANY non-zero numerical tendency is an error.

Iter-792 computes `fv3_sw_tendencies` once at t=0 on the exact W2
IC and reports where |du_d/dt| and |dv_d/dt| are largest.  If peak
tendencies localise at cube vertices, the tendency operator IS the
direct source of the W2 mode-A (accumulation over 288 timesteps
produces the observed 0.159 m/s v_ll_Linf).

Scope: observational only.  No source-code change, no new sentinel.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


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
div_damp = 8.0 * _div_damp_cube(n)

grid = create_cubed_sphere(n=n, use_duogrid=False)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=div_damp,
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

sw = williamson_test2(grid)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

# Compute t=0 tendencies once.
dh_dt, du_dt, dv_dt = fv3_sw_tendencies(
    sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
    g=cfg.g, div_damp=cfg.div_damp,
    hyperdiff_coeff=cfg.hyperdiff_coeff,
    boundary_fix=cfg.boundary_fix,
    boundary_fix_skip_corners=cfg.boundary_fix_skip_corners,
    fortran_a2b_corner_avg=cfg.fortran_a2b_corner_avg,
    fortran_vector_corner_fill=cfg.fortran_vector_corner_fill,
)

dh_dt = np.asarray(dh_dt)
du_dt = np.asarray(du_dt)
dv_dt = np.asarray(dv_dt)

lat_cc = np.rad2deg(np.asarray(grid.lat))
lon_cc = np.rad2deg(np.asarray(grid.lon))
lon_cc = np.where(lon_cc > 180.0, lon_cc - 360.0, lon_cc)

lat_u = np.rad2deg(np.asarray(cdgrid.lat_edge_x))
lon_u = np.rad2deg(np.asarray(cdgrid.lon_edge_x))
lon_u = np.where(lon_u > 180.0, lon_u - 360.0, lon_u)

lat_v = np.rad2deg(np.asarray(cdgrid.lat_edge_y))
lon_v = np.rad2deg(np.asarray(cdgrid.lon_edge_y))
lon_v = np.where(lon_v > 180.0, lon_v - 360.0, lon_v)


def _peak_report(label, field, lat, lon):
    abs_field = np.abs(field)
    idx = np.unravel_index(np.argmax(abs_field), abs_field.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    peak = float(abs_field[face, ci, cj])
    latd = float(lat[face, ci, cj])
    lond = float(lon[face, ci, cj])
    gc = _gc_dist_to_nearest_cube_vertex_deg(latd, lond)
    total_mean = float(np.mean(abs_field))
    # Count cells with |x| > 0.5 * peak and their cube-vertex prox.
    hot = abs_field > 0.5 * peak
    n_hot = int(np.sum(hot))
    n_near = 0
    n_at_pole = 0
    if n_hot > 0:
        hot_lat = lat[hot]
        hot_lon = lon[hot]
        for i in range(n_hot):
            g = _gc_dist_to_nearest_cube_vertex_deg(float(hot_lat[i]),
                                                     float(hot_lon[i]))
            if g < 10.0:
                n_near += 1
            if abs(float(hot_lat[i])) > 80.0:
                n_at_pole += 1
    print(f"  {label}  peak={peak:.3e}  face={face}  (i,j)=({ci},{cj})  "
          f"lat={latd:+.2f}°  lon={lond:+.2f}°  "
          f"GC-to-vertex={gc:.2f}°  |mean|={total_mean:.3e}  "
          f"hot={n_hot}  near_vertex={n_near}  near_pole={n_at_pole}")


print(f"Iter-792 W2 t=0 tendency cube-vertex audit at C36 (LEGACY)")
print(f"For solid-body rotation W2, analytical tendencies are all ZERO.")
print(f"Any non-zero numerical tendency is error.")
print()
print(f"{'field':>8}  {'peak':>10}  {'face,(i,j)':>15}  "
      f"{'lat':>8}  {'lon':>8}  {'GC-to-vertex':>13}  {'|mean|':>10}  "
      f"{'hot':>5}  {'near_vertex':>12}  {'near_pole':>10}")
print("-" * 130)
_peak_report("dh/dt", dh_dt, lat_cc, lon_cc)
_peak_report("du/dt", du_dt, lat_u, lon_u)
_peak_report("dv/dt", dv_dt, lat_v, lon_v)
print()
print("hot := |tendency| > 0.5 * peak")
print("near_vertex := hot cells within 10° GC of any cube vertex")
print("near_pole := hot cells with |lat| > 80°")

print()
print("Interpretation cues (observational only):")
print("- If dv/dt peaks at cube vertices (GC < 5°) with most hot")
print("  points near vertices: `_arakawa_lamb_gradient` on B has a")
print("  cube-vertex bug that causes the W2 mode-A.")
print("- If dv/dt peaks near the pole (|lat| > 80°) or mid-face")
print("  (GC > 30°): a different mechanism is at work.")
print("- dh/dt for W2 should be near machine precision (mass")
print("  conservation is exact for steady state).  Non-trivial dh/dt")
print("  would indicate transport-chain error.")

print()
print("What iter-792 DOES measure (observational only):")
print("- Peak t=0 tendency magnitudes and locations for W2 IC.")
print("- Hot-cell (|tend| > 0.5*peak) cube-vertex and pole proximity.")
print("What it does NOT establish:")
print("- Which SPECIFIC operator in fv3_sw_tendencies produces the")
print("  cube-vertex peak — that requires finer decomposition (toggle")
print("  each operator separately).")
print("- Whether the t=0 peak-tendency cells become the dominant")
print("  v_ll_Linf contributors after 288 steps of accumulation.")
