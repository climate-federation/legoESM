"""Iter-791 diagnostic: `fv3_cc2c` fidelity for the solid-body
rotation IC at C36.

Per iter-790's redirect: the W2 production D→C pipeline is
  fv3_d2cc (D-edge → cell centre, simple 2-pt average)
  fv3_cc2c (cell centre → C-grid, via pad_halo_vector + 2-pt
           average + non-orthogonality correction with cosa_u).

Iter-791 measures the cube-vertex error of (u_c, v_c) output from
`fv3_cc2c` on the analytical solid-body rotation cell-centre input.
This bypasses `fv3_d2cc` (which is a simple 2-pt average with no
physical error) and isolates `fv3_cc2c`'s behaviour.

Method:
  1. Compute cell-centre u_east, v_north analytically at each
     cell centre for β=π/4 solid-body rotation.
  2. Project to grid-axis cell-centre winds:
       u_cc = cos_angle * u_east + sin_angle * v_north
       v_cc = -sin_angle * u_east + cos_angle * v_north
  3. Call `fv3_cc2c(u_cc, v_cc, cdgrid)` → computed (u_c, v_c).
  4. Compute analytical (u_c_exact, v_c_exact) at each C-grid
     stagger from the geographic winds at those positions.
  5. Report peak |err| + GC-to-vertex.

If peak |err| in (u_c, v_c) lives at cube vertices (GC ≲ few
degrees), `fv3_cc2c` has a cube-vertex bug that directly feeds
the W2 mode-A artifact.  If peak |err| is elsewhere, `fv3_cc2c`
is not the primary W2 mode-A source and another operator
(momentum tendencies, KE gradient) must be audited.

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
from legoesm.core.operators_cdgrid import fv3_cc2c
from tests.test_cases.cosine_bell import _rotation_winds_geo


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


def _fidelity_one(use_duogrid, n, beta):
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=_div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    # Analytical cell-centre (u_cc, v_cc) in grid-axis convention.
    u_east_cc, v_north_cc = _rotation_winds_geo(
        grid.lon, grid.lat, grid.radius, beta)
    u_cc = grid.cos_angle * u_east_cc + grid.sin_angle * v_north_cc
    v_cc = -grid.sin_angle * u_east_cc + grid.cos_angle * v_north_cc

    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    u_c = np.asarray(u_c)
    v_c = np.asarray(v_c)

    # Analytical u_c at x-edge stagger (same as u_d position):
    # u_c is the CONTRAVARIANT x-component projected with sina_u.
    # Actually `fv3_cc2c` at line 1474:
    #   u_c = u_avg * sina_u - v_at_u * cosa_u
    # This is the projection of the cell-centre COVARIANT wind onto
    # the x-face NORMAL direction.  For a solid-body rotation, the
    # analytical u_c_exact at x-edge is:
    #   u_east_at_x_edge, v_north_at_x_edge  (analytical)
    #   u_cov = cos_angle_edge_x * u_east + sin_angle_edge_x * v_north
    #   v_cov = -sin_angle_edge_x * u_east + cos_angle_edge_x * v_north
    #   # non-orthogonality correction (same as in fv3_cc2c but with
    #   # exact cov values instead of interpolated):
    #   sina_u_at_edge = sqrt(1 - cosa_u^2)
    #   u_c_exact = u_cov * sina_u_at_edge - v_cov * cosa_u_at_edge
    # At x-edge positions (shape (6, n+1, n)):
    u_east_x, v_north_x = _rotation_winds_geo(
        cdgrid.lon_edge_x, cdgrid.lat_edge_x, grid.radius, beta)
    u_cov_x = (cdgrid.cos_angle_edge_x * u_east_x
               + cdgrid.sin_angle_edge_x * v_north_x)
    v_cov_x = (-cdgrid.sin_angle_edge_x * u_east_x
               + cdgrid.cos_angle_edge_x * v_north_x)
    # Wait, u_c lives at x-edge but `u_d`-shape (6, n, n+1) whereas
    # iter-791's u_c is at "x-face normal" which in FV3 convention
    # is (6, n+1, n).  Let me check the shape.

    # u_c shape from fv3_cc2c is (6, n+1, n).
    # cdgrid.cosa_u has shape (6, n+1, n), cdgrid.sina_u derived.
    cosa_u = np.asarray(cdgrid.cosa_u)
    sina_u = np.sqrt(np.maximum(1.0 - cosa_u ** 2, 1e-30))

    # u_c at "x-face" (i-edge, j-cell) stagger — but that's shape
    # (6, n+1, n).  We need analytical u_c at that stagger.  The
    # lat/lon at that stagger — which cdgrid field gives it?

    # Looking at cdgrid.lon_edge_y / lat_edge_y shapes (6, n+1, n):
    # these are at "y-edge" in the CD-grid naming.  Let me verify by
    # shape match.  cdgrid.cosa_u is (6, n+1, n), and cdgrid.lat_edge_y
    # is also (6, n+1, n).  So cosa_u lives at lat_edge_y positions.
    # Therefore u_c (same shape) is at the lat_edge_y stagger —
    # which is the cell x-face position in the CD-grid convention
    # (i-edge, j-cell).
    u_east_at_uc, v_north_at_uc = _rotation_winds_geo(
        cdgrid.lon_edge_y, cdgrid.lat_edge_y, grid.radius, beta)
    # Covariant at that stagger:
    u_cov_uc = (cdgrid.cos_angle_edge_y * u_east_at_uc
                + cdgrid.sin_angle_edge_y * v_north_at_uc)
    v_cov_uc = (-cdgrid.sin_angle_edge_y * u_east_at_uc
                + cdgrid.cos_angle_edge_y * v_north_at_uc)
    # Analytical u_c via the same formula as fv3_cc2c (line 1474):
    u_c_exact = np.asarray(u_cov_uc) * sina_u - np.asarray(v_cov_uc) * cosa_u

    # v_c is at "y-face" (i-cell, j-edge) stagger, shape (6, n, n+1).
    # The fv3_cc2c formula (line 1476): v_c = 0.5 * (v_pad[j-1] + v_pad[j])
    # — no correction (y-edge is along e_perp which IS the y-face normal).
    # So v_c_exact = v_cov at cell-centre, averaged to j-edge.
    # Actually simpler: v_c_exact is the v-component along the grid e_perp
    # direction at (i, j+1/2).  For edge-midpoint stagger, v at cell i
    # cell j+1/2 = v_cov at lat_edge_x (which is the x-edge in FV3
    # parlance; shape (6, n, n+1)).
    u_east_at_vc, v_north_at_vc = _rotation_winds_geo(
        cdgrid.lon_edge_x, cdgrid.lat_edge_x, grid.radius, beta)
    v_cov_vc = (-cdgrid.sin_angle_edge_x * u_east_at_vc
                + cdgrid.cos_angle_edge_x * v_north_at_vc)
    v_c_exact = np.asarray(v_cov_vc)

    u_c_err = np.abs(u_c - u_c_exact)
    v_c_err = np.abs(v_c - v_c_exact)

    lat_u = np.rad2deg(np.asarray(cdgrid.lat_edge_y))
    lon_u = np.rad2deg(np.asarray(cdgrid.lon_edge_y))
    lon_u = np.where(lon_u > 180.0, lon_u - 360.0, lon_u)
    lat_v = np.rad2deg(np.asarray(cdgrid.lat_edge_x))
    lon_v = np.rad2deg(np.asarray(cdgrid.lon_edge_x))
    lon_v = np.where(lon_v > 180.0, lon_v - 360.0, lon_v)

    def _peak(label, err, lat, lon):
        idx = np.unravel_index(np.argmax(err), err.shape)
        face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
        peak = float(err[face, ci, cj])
        latd = float(lat[face, ci, cj])
        lond = float(lon[face, ci, cj])
        gc = _gc_dist_to_nearest_cube_vertex_deg(latd, lond)
        return {
            'peak': peak, 'face': face, 'ci': ci, 'cj': cj,
            'lat': latd, 'lon': lond, 'gc': gc,
        }

    u_c_peak = _peak("u_c", u_c_err, lat_u, lon_u)
    v_c_peak = _peak("v_c", v_c_err, lat_v, lon_v)

    u_c_scale = float(np.max(np.abs(u_c_exact)))
    v_c_scale = float(np.max(np.abs(v_c_exact)))

    return {
        'u_c_peak': u_c_peak, 'v_c_peak': v_c_peak,
        'u_c_scale': u_c_scale, 'v_c_scale': v_c_scale,
    }


n = 36
beta = jnp.pi / 4.0

print(f"Iter-791 `fv3_cc2c` fidelity at C36, β=π/4 solid-body rotation")
print()

for use_dg, label in ((False, 'LEGACY (use_duogrid=False)'),
                       (True,  'DUOGRID (use_duogrid=True)')):
    r = _fidelity_one(use_dg, n, beta)
    u = r['u_c_peak']; v = r['v_c_peak']
    ur = 100.0 * u['peak'] / r['u_c_scale'] if r['u_c_scale'] > 0 else np.nan
    vr = 100.0 * v['peak'] / r['v_c_scale'] if r['v_c_scale'] > 0 else np.nan
    print(f"{label}:")
    print(f"  u_c peak={u['peak']:.3e}  face={u['face']}  (i,j)=({u['ci']},{u['cj']})  "
          f"lat={u['lat']:+.2f}°  lon={u['lon']:+.2f}°  GC={u['gc']:.2f}°  "
          f"rel={ur:.2f}%")
    print(f"  v_c peak={v['peak']:.3e}  face={v['face']}  (i,j)=({v['ci']},{v['cj']})  "
          f"lat={v['lat']:+.2f}°  lon={v['lon']:+.2f}°  GC={v['gc']:.2f}°  "
          f"rel={vr:.2f}%")
    print()

print("Interpretation cues (observational only):")
print("- If peak |err| in u_c/v_c lives at GC ≲ 2°: fv3_cc2c has a")
print("  cube-vertex bug that directly feeds the W2 mode-A artifact.")
print("- If peak |err| is mid-face: fv3_cc2c is not the W2 mode-A")
print("  primary source and another operator must be audited.")

print()
print("What iter-791 DOES measure (observational only):")
print("- fv3_cc2c's peak |u_c - u_c_exact| and |v_c - v_c_exact|")
print("  on the analytical solid-body rotation cell-centre IC.")
print("What it does NOT establish:")
print("- Whether the measured fv3_cc2c error is large enough to")
print("  explain v_ll_Linf ≈ 0.159 m/s on the actual W2 run.")
print("- Which part of fv3_cc2c (pad_halo_vector, averaging,")
print("  non-orthogonality correction) contributes most.")
