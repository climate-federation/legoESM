"""Iter-853 diagnostic: Phase 1 trial wire-in (t=0 only).

Per iter-852's iter-853+ priority.  Tests whether replacing the
production cell-centre `cgrid_divergence` with a Fortran-faithful
divergence (Fortran corner delpc → 4-pt average to cell centres,
construction from iter-851) changes the t=0 W2 LEGACY |dv/dt| peak
by a manageable amount.

This is a SAFETY CHECK before a 1-day trial.  If the Phase 1 swap
explodes the t=0 peak (>10× growth), the production integration
would almost certainly destabilise.  If the peak stays within
~2× of baseline, the full 1-day wire-in is plausible.

Method:
  1. Compute baseline dv/dt at t=0 W2 IC via the production
     `fv3_sw_tendencies()`.
  2. Monkey-patch `cgrid_divergence` to return Fortran-faithful
     Fortran-cc divergence (constructed from
     `_d_sw5_corner_divergence` at corners, 4-pt avg to centres).
     The monkey-patch captures u_d, v_d, ua, va, dt at the call
     site via thread-local state to plumb them into the helper.
  3. Re-compute dv/dt with the patched divergence.
  4. Compare peak |dv/dt| and adaptive_coeff at the cube-vertex-
     adjacent peak location.

Caveats (inherited from iter-852):
- Fortran-cc is itself a constructed quantity.
- `_d_sw5_corner_divergence` nord=0 still uses mode='edge' halo for
  vort/ptc.
- The monkey-patch keeps all downstream code unchanged (A-L gradient,
  interp_corner_to_center) so iter-849 Checks 3+4 are NOT addressed.

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
from legoesm.core import operators_cdgrid as ocd
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
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
        "peak_abs": float(a[f, i, j]),
        "value": float(arr[f, i, j]),
        "lat_deg": float(np.rad2deg(lat_c)),
        "lon_deg": float(np.rad2deg(lon_c)),
        "gc_deg": _gc_deg(lat_c, lon_c),
    }


def main():
    n = 36
    dt = 300.0
    div_damp = 8.0 * 1.5e7 * (48.0 / n) ** 2  # iter-761 canonical

    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=div_damp,
        boundary_fix=True, damp_v=0.06, nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    # Closure-captured u_d/v_d/dt for the patched cgrid_divergence.
    # The monkey-patch needs access to u_d, v_d, dt to call
    # `_d_sw5_corner_divergence`.  We pre-compute ua, va via
    # `_d2a2c_vect` and bind everything in a closure.
    ua, va, _, _, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)
    da_min_c = float(jnp.min(cdgrid.area_corner))

    orig_cgrid_divergence = ocd.cgrid_divergence

    def patched_cgrid_divergence(u_c, v_c, cdg):
        # Compute Fortran-faithful corner delpc.
        ke_damping = _d_sw5_corner_divergence(
            u_d, v_d, ua, va, cdg, dt,
            d2_bg=1.0, dddmp=0.0, nord=0,
        )
        delpc_corner = ke_damping / da_min_c   # (6, n+1, n+1)
        # 4-pt corner→cc average.
        delpc_cc = 0.25 * (
            delpc_corner[:, :-1, :-1] + delpc_corner[:, 1:, :-1]
            + delpc_corner[:, :-1, 1:] + delpc_corner[:, 1:, 1:])  # (6, n, n)
        return delpc_cc

    # ----- Baseline: production cgrid_divergence -----
    _, _, dv_baseline = fv3_sw_tendencies(
        sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
        g=9.80616, div_damp=div_damp, hyperdiff_coeff=0.0,
        boundary_fix=True, boundary_fix_skip_corners=False,
        fortran_dir_aware_corners=False, fortran_a2b_corner_avg=False,
        fortran_vector_corner_fill=False, zero_mean_correction=False,
    )
    baseline = _peak(dv_baseline, cdgrid.lat_edge_y, cdgrid.lon_edge_y)

    # ----- Patched: Fortran-cc cgrid_divergence -----
    try:
        ocd.cgrid_divergence = patched_cgrid_divergence
        _, _, dv_patched = fv3_sw_tendencies(
            sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
            g=9.80616, div_damp=div_damp, hyperdiff_coeff=0.0,
            boundary_fix=True, boundary_fix_skip_corners=False,
            fortran_dir_aware_corners=False, fortran_a2b_corner_avg=False,
            fortran_vector_corner_fill=False, zero_mean_correction=False,
        )
    finally:
        ocd.cgrid_divergence = orig_cgrid_divergence

    patched = _peak(dv_patched, cdgrid.lat_edge_y, cdgrid.lon_edge_y)

    print(f"Iter-853 Phase 1 trial (t=0 only) — Fortran-cc div_field swap")
    print(f"W2 IC, C{n}, LEGACY, iter-761 canonical, dt={dt}s")
    print()
    print("Baseline production (cgrid_divergence cell-centre flux-form):")
    print(f"  peak |dv/dt|            = {baseline['peak_abs']:.3e} m/s²")
    print(f"  signed dv/dt at peak    = {baseline['value']:.3e} m/s²")
    print(f"  at face={baseline['face']} (i,j)=({baseline['i']},"
          f"{baseline['j']}), lat={baseline['lat_deg']:.2f}°, "
          f"lon={baseline['lon_deg']:.2f}°, GC={baseline['gc_deg']:.2f}°")
    print()
    print("Patched (Fortran corner delpc → 4-pt avg to cell centre):")
    print(f"  peak |dv/dt|            = {patched['peak_abs']:.3e} m/s²")
    print(f"  signed dv/dt at peak    = {patched['value']:.3e} m/s²")
    print(f"  at face={patched['face']} (i,j)=({patched['i']},"
          f"{patched['j']}), lat={patched['lat_deg']:.2f}°, "
          f"lon={patched['lon_deg']:.2f}°, GC={patched['gc_deg']:.2f}°")
    print()
    if baseline['peak_abs'] > 0:
        ratio = patched['peak_abs'] / baseline['peak_abs']
        print(f"Peak ratio (patched / baseline) = {ratio:.3f}")
    print()
    print("Stability heuristic (observation only):")
    print("- ratio < 2:  full 1-day wire-in is plausible.")
    print("- 2 < ratio < 10:  full 1-day wire-in is risky; needs damp_v")
    print("                   or div_damp re-tuning to maintain stability.")
    print("- ratio > 10: 1-day wire-in would almost certainly destabilise")
    print("              without significant additional changes (e.g.,")
    print("              concurrently fixing iter-849 Checks 1+3+4).")


if __name__ == "__main__":
    main()
