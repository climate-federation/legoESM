"""Iter-850 diagnostic: compare divergence stencils at t=0 W2 IC.

Per iter-849's Priority 2 architecture audit: the current A-L
production path uses `cgrid_divergence` (cell-centre flux-form
stencil at `operators_cdgrid.py:420-441`) + 4-point centre→corner
average for div_damp.  Fortran d_sw5 uses an edge-by-edge
ptc/vort stencil with metric weights (sw_core.F90:1644-1707) to
compute corner `delpc` directly.

iter-850 measures the difference between the two stencils at
t=0 on the W2 IC, focusing on the cube-vertex-adjacent corner
where iter-848 showed the production |dv/dt| peaks.

Method:
  1. Compute `div_centre = cgrid_divergence(u_c, v_c, cdgrid)` —
     the current production cell-centre divergence.
  2. Average `div_centre` to corners via the 4-point centre→corner
     stencil (matches the iter-844 diag's projection).
  3. Compute `delpc_corner = ke_damping_from_d_sw5_corner_div(
       u_d, v_d, ua, va, cdgrid, dt, d2_bg=0, dddmp=0.2)` —
     Fortran-faithful corner delpc via the existing
     `_d_sw5_corner_divergence` helper.
  4. Report: peak |div_centre_at_corner|, peak |delpc_corner|,
     peak locations + GC-to-cube-vertex; |Δ| at cube-vertex-
     adjacent corners.

This informs whether Phase 1 of the d_sw5 architectural port (swap
`cgrid_divergence` for `_d_sw5_corner_divergence`) would
materially change the cube-vertex damping signal.

Observational only.  No production source-code change.
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

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import (
    cgrid_divergence, fv3_d2cc, fv3_cc2c)
from legoesm.core.fv3_sw_core import (
    _d2a2c_vect, _d_sw5_corner_divergence)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_deg(lat_rad, lon_rad):
    best = np.inf
    for vlat in CUBE_VERTEX_LATS:
        for vlon in CUBE_VERTEX_LONS:
            dlat = lat_rad - vlat
            dlon = lon_rad - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_rad) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            best = min(best, np.rad2deg(2.0 * np.arcsin(np.sqrt(max(a, 0.0)))))
    return float(best)


def _peak(arr, lat, lon):
    a = np.abs(np.asarray(arr))
    flat = a.reshape(6, -1)
    f = int(np.argmax(np.max(flat, axis=1)))
    ij = int(np.argmax(flat[f]))
    n_p = arr.shape[2]
    i, j = ij // n_p, ij % n_p
    lat_c = float(np.asarray(lat)[f, i, j])
    lon_c = float(np.asarray(lon)[f, i, j])
    return {
        "face": f, "i": i, "j": j,
        "peak": float(a[f, i, j]),
        "lat_deg": float(np.rad2deg(lat_c)),
        "lon_deg": float(np.rad2deg(lon_c)),
        "gc_deg": _gc_deg(lat_c, lon_c),
    }


def main():
    n = 36
    dt = 300.0

    grid = create_cubed_sphere(n=n, use_duogrid=False)
    div_damp = 8.0 * 1.5e7 * (48.0 / n) ** 2
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=div_damp,
        boundary_fix=True, damp_v=0.06, nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    # ----- Production cgrid_divergence (cell centres) -----
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    div_centre = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n)

    # 4-point centre→corner average (matches the iter-844 diag's
    # `_interp_corner_to_center` inverse direction): pad cell-centre
    # field by 1 halo, average 2x2 to get corner field of shape
    # (6, n+1, n+1).
    div_centre_pad = jnp.pad(div_centre, [(0, 0), (1, 1), (1, 1)],
                              mode='edge')
    div_centre_corner = 0.25 * (
        div_centre_pad[:, :-1, :-1] + div_centre_pad[:, 1:, :-1]
        + div_centre_pad[:, :-1, 1:] + div_centre_pad[:, 1:, 1:])  # (6, n+1, n+1)

    # ----- Fortran-faithful corner delpc via _d_sw5_corner_divergence -----
    # The helper returns the KE-damping increment, NOT delpc directly.
    # For an apples-to-apples comparison we want the raw delpc — so call
    # with d2_bg = 1 and dddmp = 0 to get a coefficient = da_min_c
    # (constant) and recover delpc as ke_damping / da_min_c.
    ua, va, _, _, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)
    da_min_c = float(jnp.min(cdgrid.area_corner))
    ke_damping = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, nord=0,
    )
    delpc_corner = np.asarray(ke_damping) / da_min_c  # (6, n+1, n+1)

    # ----- Compare at corner positions -----
    lat_corner = cdgrid.lat_corner
    lon_corner = cdgrid.lon_corner

    centre_peak = _peak(div_centre_corner, lat_corner, lon_corner)
    delpc_peak = _peak(delpc_corner, lat_corner, lon_corner)

    print(f"Iter-850 divergence stencil comparison at t=0 W2 IC, C{n}, "
          f"LEGACY")
    print()
    print(f"Production stencil (cell-centre cgrid_divergence + 4-pt avg):")
    print(f"  peak |div_centre_corner| = {centre_peak['peak']:.3e} 1/s")
    print(f"  at face={centre_peak['face']} (i,j)=({centre_peak['i']},"
          f"{centre_peak['j']}), lat={centre_peak['lat_deg']:.2f}°, "
          f"lon={centre_peak['lon_deg']:.2f}°, GC={centre_peak['gc_deg']:.2f}°")
    print()
    print(f"Fortran-faithful stencil (_d_sw5_corner_divergence delpc):")
    print(f"  peak |delpc_corner|      = {delpc_peak['peak']:.3e} 1/s")
    print(f"  at face={delpc_peak['face']} (i,j)=({delpc_peak['i']},"
          f"{delpc_peak['j']}), lat={delpc_peak['lat_deg']:.2f}°, "
          f"lon={delpc_peak['lon_deg']:.2f}°, GC={delpc_peak['gc_deg']:.2f}°")
    print()
    delta = np.asarray(delpc_corner) - np.asarray(div_centre_corner)
    delta_peak = _peak(delta, lat_corner, lon_corner)
    print(f"Stencil delta (Fortran − production):")
    print(f"  peak |Δ|                 = {delta_peak['peak']:.3e} 1/s")
    print(f"  at face={delta_peak['face']} (i,j)=({delta_peak['i']},"
          f"{delta_peak['j']}), GC-to-vertex={delta_peak['gc_deg']:.2f}°")
    print()
    if centre_peak['peak'] > 0:
        global_ratio = delpc_peak['peak'] / centre_peak['peak']
        print(f"GLOBAL peak-vs-peak ratio Fortran/production = {global_ratio:.3f}")
        print(f"  (NOTE: peaks are at DIFFERENT locations — see GC values above)")

    # Same-location ratios (Codex iter-850b request).  Compare at the
    # production peak location and at the Fortran peak location to
    # disentangle magnitude difference from location difference.
    print()
    print("Same-location ratios (Fortran/production):")
    f_p, i_p, j_p = centre_peak["face"], centre_peak["i"], centre_peak["j"]
    prod_at_prod_peak = float(np.asarray(div_centre_corner)[f_p, i_p, j_p])
    fortran_at_prod_peak = float(np.asarray(delpc_corner)[f_p, i_p, j_p])
    if abs(prod_at_prod_peak) > 0:
        ratio_at_prod = fortran_at_prod_peak / prod_at_prod_peak
        print(f"  AT production peak (face {f_p} ({i_p},{j_p}), GC={centre_peak['gc_deg']:.2f}°):")
        print(f"    production = {prod_at_prod_peak:.3e},"
              f" Fortran = {fortran_at_prod_peak:.3e},"
              f" ratio = {ratio_at_prod:.3f}")
    f_f, i_f, j_f = delpc_peak["face"], delpc_peak["i"], delpc_peak["j"]
    prod_at_fortran_peak = float(np.asarray(div_centre_corner)[f_f, i_f, j_f])
    fortran_at_fortran_peak = float(np.asarray(delpc_corner)[f_f, i_f, j_f])
    if abs(prod_at_fortran_peak) > 0:
        ratio_at_fortran = fortran_at_fortran_peak / prod_at_fortran_peak
        print(f"  AT Fortran peak (face {f_f} ({i_f},{j_f}), GC={delpc_peak['gc_deg']:.2f}°):")
        print(f"    production = {prod_at_fortran_peak:.3e},"
              f" Fortran = {fortran_at_fortran_peak:.3e},"
              f" ratio = {ratio_at_fortran:.3f}")
    else:
        print(f"  AT Fortran peak (face {f_f} ({i_f},{j_f}), GC={delpc_peak['gc_deg']:.2f}°):")
        print(f"    production = {prod_at_fortran_peak:.3e} (zero, ratio undefined),"
              f" Fortran = {fortran_at_fortran_peak:.3e}")
    print()
    print("Caveat (Codex iter-850b note):")
    print("- The Fortran `_d_sw5_corner_divergence` nord=0 branch still uses")
    print("  `mode='edge'` padding for vort/ptc at fv3_sw_core.py:~1044, an")
    print("  acknowledged same-face halo gap.  The peak at GC=35.26° (panel-")
    print("  edge equatorial corner) could therefore reflect that halo gap")
    print("  rather than a true Fortran-faithful stencil signature.  A complete")
    print("  comparison would require fixing that halo gap first.")
    print()
    print("Interpretation (observational only):")
    print("- If Δ peaks AT cube-vertex corners with magnitude comparable")
    print("  to the production divergence: stencil swap is meaningful and")
    print("  Phase 1 of the d_sw5 port could materially change the cube-")
    print("  vertex damping signal.")
    print("- If Δ is small everywhere: the stencil swap is cosmetic; the")
    print("  cube-vertex amplification comes from elsewhere in the d_sw5")
    print("  pipeline (probably the application path — Check 4 in iter-849).")


if __name__ == "__main__":
    main()
