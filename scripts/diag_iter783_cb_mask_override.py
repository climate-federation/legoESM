"""Iter-783 diagnostic: targeted override of `_d2a2c_vect`'s
cube-vertex error on the cosine bell run.

Iter-782 measured that `_d2a2c_vect` on the analytical solid-body
rotation IC has ~6 m/s (≈ 11% relative) error in (ut, vt)
localised at the 8 cube vertices.  The error is imprinted on the
transport winds for the whole 1-day cosine bell run.

Iter-783 tests whether that measured error actually CAUSES the
cosine bell distortion at t=1 day.  Method:
  1. Run the cosine bell 1-day at C36 with unmodified `_d2a2c_vect`
     output → BASELINE Linf.
  2. Re-run with (ut, vt) replaced by their ANALYTICAL values at
     every C-grid stagger cell whose GC-distance to any cube
     vertex is ≤ MASK_RADIUS (tests at 5°, 10°, 20°).
  3. Compare Linf across baseline + three override runs.

If override reduces Linf significantly (say, ≥ 50%), the
`_d2a2c_vect` cube-vertex error is a primary driver of the
cosine bell distortion.  If Linf is unchanged or only weakly
reduced, the mechanism is NOT primary and other sources (PPM
limiter, fv_tp_2d edge handling, integrated face-count error)
must be considered.

Scope: observational only, no source-code change, no new sentinel.
The override is a DIAGNOSTIC-ONLY patch applied outside the
production path.
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
    cosine_bell_cubesphere, cosine_bell_exact, _rotation_winds_geo)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_mask(lat_r, lon_r, max_rad):
    """Boolean mask: True where GC-distance to ANY cube vertex ≤ max_rad."""
    mask = np.zeros_like(lat_r, dtype=bool)
    for vlat in CUBE_VERTEX_LATS_R:
        for vlon in CUBE_VERTEX_LONS_R:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            mask |= (d <= max_rad)
    return mask


def _run(n, dt, days, mask_radius_deg=None):
    """Run cosine bell; optionally override ut/vt near cube vertices
    with the analytical solid-body rotation values.

    Parameters
    ----------
    mask_radius_deg : float or None
        If None, use unmodified (ut, vt) from `_d2a2c_vect` (baseline).
        Else, replace (ut, vt) with analytical at every stagger cell
        within `mask_radius_deg` GC-distance of any cube vertex.
    """
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
    ua, va, uc, vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)

    if mask_radius_deg is not None:
        max_rad = np.deg2rad(mask_radius_deg)

        # Analytical ut at y-edge stagger (same shape as ut).
        u_e_y, v_n_y = _rotation_winds_geo(
            cdgrid.lon_edge_y, cdgrid.lat_edge_y, grid.radius, beta)
        u_cov_y = (cdgrid.cos_angle_edge_y * u_e_y
                   + cdgrid.sin_angle_edge_y * v_n_y)
        v_cov_y = (-cdgrid.sin_angle_edge_y * u_e_y
                   + cdgrid.cos_angle_edge_y * v_n_y)
        ut_exact = (u_cov_y - v_cov_y * cdgrid.cosa_u) * cdgrid.rsin_u

        # Analytical vt at x-edge stagger (same shape as vt).
        u_e_x, v_n_x = _rotation_winds_geo(
            cdgrid.lon_edge_x, cdgrid.lat_edge_x, grid.radius, beta)
        u_cov_x = (cdgrid.cos_angle_edge_x * u_e_x
                   + cdgrid.sin_angle_edge_x * v_n_x)
        v_cov_x = (-cdgrid.sin_angle_edge_x * u_e_x
                   + cdgrid.cos_angle_edge_x * v_n_x)
        vt_exact = (v_cov_x - u_cov_x * cdgrid.cosa_v) * cdgrid.rsin_v

        ut_mask = _gc_mask(
            np.asarray(cdgrid.lat_edge_y),
            np.asarray(cdgrid.lon_edge_y),
            max_rad)
        vt_mask = _gc_mask(
            np.asarray(cdgrid.lat_edge_x),
            np.asarray(cdgrid.lon_edge_x),
            max_rad)

        ut = jnp.where(ut_mask, ut_exact, ut)
        vt = jnp.where(vt_mask, vt_exact, vt)

        n_ut_override = int(np.sum(ut_mask))
        n_vt_override = int(np.sum(vt_mask))
    else:
        n_ut_override = 0
        n_vt_override = 0

    mass_init = float(jnp.sum(state.h * grid.area))
    h = state.h
    for _ in range(n_steps):
        h = transport_step(h, ut, vt, dt_exact, cdgrid,
                            mass_target=mass_init)

    t_s = days * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    err = np.asarray(h) - np.asarray(h_exact)
    abs_err = np.abs(err)

    l_inf = float(np.max(abs_err))
    area_np = np.asarray(grid.area)
    h_exact_np = np.asarray(h_exact)
    l2_num = float(np.sum(area_np * err ** 2))
    l2_den = float(np.sum(area_np * h_exact_np ** 2))
    l2 = float(np.sqrt(l2_num / l2_den)) if l2_den > 0 else np.nan

    # Peak-error location.
    idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    lat = float(np.rad2deg(np.asarray(grid.lat)[face, ci, cj]))
    lon = float(np.rad2deg(np.asarray(grid.lon)[face, ci, cj]))
    if lon > 180: lon -= 360

    return {
        'l_inf': l_inf,
        'l2': l2,
        'peak_face': face,
        'peak_cell': (ci, cj),
        'peak_lat': lat,
        'peak_lon': lon,
        'n_ut_override': n_ut_override,
        'n_vt_override': n_vt_override,
    }


n = 36
dt = 1440.0  # time-aligned with iter-781b
days = 1.0

print(f"Iter-783 cosine bell C36 with targeted ut/vt override at cube vertices")
print(f"dt={dt}s, β=π/4, {days} day")
print()
print(f"{'case':>30}  {'L_inf':>9}  {'L2':>9}  {'peak':>14}  "
      f"{'overrides':>20}")
print("-" * 95)

baseline = _run(n, dt, days, mask_radius_deg=None)
print(f"{'BASELINE (no override)':>30}  {baseline['l_inf']:>9.3e}  "
      f"{baseline['l2']:>9.3e}  "
      f"{'face '+str(baseline['peak_face'])+' ('+str(baseline['peak_cell'][0])+','+str(baseline['peak_cell'][1])+')':>14}  "
      f"{'ut=0, vt=0':>20}")

for r_deg in (5.0, 10.0, 20.0):
    result = _run(n, dt, days, mask_radius_deg=r_deg)
    delta_linf = 100.0 * (result['l_inf'] - baseline['l_inf']) / baseline['l_inf']
    ov_s = f"ut={result['n_ut_override']}, vt={result['n_vt_override']}"
    print(f"{'override GC <= '+str(r_deg)+'°':>30}  {result['l_inf']:>9.3e}  "
          f"{result['l2']:>9.3e}  "
          f"{'face '+str(result['peak_face'])+' ('+str(result['peak_cell'][0])+','+str(result['peak_cell'][1])+')':>14}  "
          f"{ov_s:>20}"
          f"  [{delta_linf:+.1f}% Linf]")

print()
print("What iter-783 DOES measure (observational only):")
print("- 1-day cosine bell Linf and L2 under BASELINE (unmodified")
print("  _d2a2c_vect) vs OVERRIDE (analytical ut/vt at cells within")
print("  GC <= {5,10,20}° of any cube vertex).")
print("- Count of stagger cells replaced at each mask radius.")
print("What it does NOT establish:")
print("- Whether the override is PHYSICALLY CONSISTENT (the override")
print("  breaks the local covariant/contravariant consistency with")
print("  the unmodified h-field and boundary_fix stabiliser); large")
print("  Linf reductions may partly reflect artificial smoothness")
print("  imposed by the override.")
print("- Which PART of `_d2a2c_vect` (fill_corner_region vs A→C")
print("  stencil) is responsible for the measured error — a separate")
print("  iter (iter-784+) would need to toggle each part.")
print("- Whether fixing the cube-vertex error in `_d2a2c_vect` itself")
print("  (via a Fortran-faithful change) would give the same Linf")
print("  reduction as the override; the override is a diagnostic-only")
print("  analytical injection, not a production fix.")
