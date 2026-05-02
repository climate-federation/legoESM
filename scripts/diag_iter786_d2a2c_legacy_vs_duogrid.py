"""Iter-786 diagnostic: `_d2a2c_vect` legacy vs duogrid path
fidelity comparison at C36.

Iter-782 measured `_d2a2c_vect` cube-vertex error at 11.69%
relative on the solid-body rotation IC at C36.  That measurement
was taken with `create_cubed_sphere(n)` using the default
`use_duogrid=False` — so it measured the LEGACY (non-duogrid)
path, not the duogrid path.

The production W2 sentinels (`test_w2_alpha0_c36_1day_iter761_
matrix_config`) ALSO use `use_duogrid=False`, so they exercise
the same LEGACY path.  The W2 mode-A artifact at v_ll_Linf ≈
0.159 m/s localises at 8 cube vertices — which is exactly where
iter-782's 11% error in `_d2a2c_vect` lives.

Iter-786 tests whether the duogrid path has SMALLER cube-vertex
error than the legacy path.  If yes, switching W2 to
`use_duogrid=True` could resolve the W2 artifact.  If no, the
cube-vertex error is in a deeper layer (metrics, halo, etc.)
that the duogrid switch doesn't help.

Method:
  1. Build two grids: `create_cubed_sphere(n, use_duogrid=False)`
     (LEGACY) and `create_cubed_sphere(n, use_duogrid=True)`
     (DUOGRID).  Otherwise identical (same n, same radius).
  2. Create cdgrid + state from each; call `_d2a2c_vect` on each;
     compute ut_exact / vt_exact analytically at each stagger.
  3. Report peak |err| + location + GC-to-vertex + relative error
     for both paths side-by-side.

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


def _run_one(n, use_duogrid, beta):
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
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
    ut = np.asarray(ut)
    vt = np.asarray(vt)

    # Analytical ut at y-edge stagger.
    u_e_y, v_n_y = _rotation_winds_geo(
        cdgrid.lon_edge_y, cdgrid.lat_edge_y, grid.radius, beta)
    u_cov_y = (cdgrid.cos_angle_edge_y * u_e_y
               + cdgrid.sin_angle_edge_y * v_n_y)
    v_cov_y = (-cdgrid.sin_angle_edge_y * u_e_y
               + cdgrid.cos_angle_edge_y * v_n_y)
    ut_exact = (u_cov_y - v_cov_y * cdgrid.cosa_u) * cdgrid.rsin_u
    ut_exact = np.asarray(ut_exact)

    u_e_x, v_n_x = _rotation_winds_geo(
        cdgrid.lon_edge_x, cdgrid.lat_edge_x, grid.radius, beta)
    u_cov_x = (cdgrid.cos_angle_edge_x * u_e_x
               + cdgrid.sin_angle_edge_x * v_n_x)
    v_cov_x = (-cdgrid.sin_angle_edge_x * u_e_x
               + cdgrid.cos_angle_edge_x * v_n_x)
    vt_exact = (v_cov_x - u_cov_x * cdgrid.cosa_v) * cdgrid.rsin_v
    vt_exact = np.asarray(vt_exact)

    ut_err = np.abs(ut - ut_exact)
    vt_err = np.abs(vt - vt_exact)

    lat_y = np.rad2deg(np.asarray(cdgrid.lat_edge_y))
    lon_y = np.rad2deg(np.asarray(cdgrid.lon_edge_y))
    lon_y = np.where(lon_y > 180.0, lon_y - 360.0, lon_y)
    lat_x = np.rad2deg(np.asarray(cdgrid.lat_edge_x))
    lon_x = np.rad2deg(np.asarray(cdgrid.lon_edge_x))
    lon_x = np.where(lon_x > 180.0, lon_x - 360.0, lon_x)

    dg = grid.duogrid
    dg_info = f"None" if dg is None else f"ng={dg.ng}"
    print(f"  grid.duogrid: {dg_info}")

    def _peak(label, err, lat, lon):
        idx = np.unravel_index(np.argmax(err), err.shape)
        face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
        peak = float(err[face, ci, cj])
        latd = float(lat[face, ci, cj])
        lond = float(lon[face, ci, cj])
        gc = _gc_dist_to_nearest_cube_vertex_deg(latd, lond)
        return peak, face, ci, cj, latd, lond, gc

    ut_peak, uf, ui, uj, ul, uo, ug = _peak("ut", ut_err, lat_y, lon_y)
    vt_peak, vf, vi, vj, vl, vo, vg = _peak("vt", vt_err, lat_x, lon_x)

    ut_scale = float(np.max(np.abs(ut_exact)))
    vt_scale = float(np.max(np.abs(vt_exact)))
    ut_rel = 100.0 * ut_peak / ut_scale if ut_scale > 0 else np.nan
    vt_rel = 100.0 * vt_peak / vt_scale if vt_scale > 0 else np.nan

    print(f"  ut  peak={ut_peak:.3e}  face={uf}  (i,j)=({ui},{uj})  "
          f"lat={ul:+.2f}°  lon={uo:+.2f}°  GC={ug:.2f}°  rel={ut_rel:.2f}%")
    print(f"  vt  peak={vt_peak:.3e}  face={vf}  (i,j)=({vi},{vj})  "
          f"lat={vl:+.2f}°  lon={vo:+.2f}°  GC={vg:.2f}°  rel={vt_rel:.2f}%")
    return {
        'ut_peak': ut_peak, 'vt_peak': vt_peak,
        'ut_rel': ut_rel, 'vt_rel': vt_rel,
        'ut_gc': ug, 'vt_gc': vg,
    }


n = 36
beta = jnp.pi / 4.0

print(f"Iter-786 `_d2a2c_vect` legacy vs duogrid path, C36, β=π/4")
print()
print(f"LEGACY path (use_duogrid=False):")
legacy = _run_one(n, use_duogrid=False, beta=beta)
print()
print(f"DUOGRID path (use_duogrid=True):")
duogrid = _run_one(n, use_duogrid=True, beta=beta)

print()
print(f"Side-by-side comparison of peak |err|:")
print(f"{'':>10}  {'LEGACY':>12}  {'DUOGRID':>12}  {'ratio (D/L)':>14}")
for label, l, d in (('ut peak', legacy['ut_peak'], duogrid['ut_peak']),
                     ('vt peak', legacy['vt_peak'], duogrid['vt_peak']),
                     ('ut rel %', legacy['ut_rel'], duogrid['ut_rel']),
                     ('vt rel %', legacy['vt_rel'], duogrid['vt_rel'])):
    ratio = d / l if l > 0 else np.nan
    print(f"  {label:>8}  {l:>12.3e}  {d:>12.3e}  {ratio:>14.3f}")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID peaks are MUCH SMALLER than LEGACY (ratio < 0.5):")
print("  switching W2 to `use_duogrid=True` could reduce the W2")
print("  mode-A artifact at cube vertices.")
print("- If DUOGRID peaks are SIMILAR to LEGACY (ratio ≈ 1):")
print("  the cube-vertex error is in a deeper layer than the halo")
print("  exchange choice; duogrid switch does not help.")
print("- If DUOGRID peaks are LARGER (ratio > 1): the duogrid path")
print("  introduces its own cube-vertex error source.")

print()
print("What iter-786 DOES measure (observational only):")
print("- `_d2a2c_vect` peak |err| at cube vertices for legacy vs")
print("  duogrid halo paths, at C36 on the solid-body rotation IC.")
print("What it does NOT establish:")
print("- Whether switching W2 to `use_duogrid=True` actually reduces")
print("  the W2 mode-A artifact at v_ll_Linf ≈ 0.159 m/s — that")
print("  requires running a W2 integration under both paths, which")
print("  iter-786 does not do.")
print("- Whether the duogrid path changes OTHER parts of the")
print("  shallow-water pipeline in ways that affect W2 accuracy.")
