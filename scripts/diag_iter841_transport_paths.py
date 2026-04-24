"""Iter-841 diagnostic: compare dh/dt from the two candidate
transport paths on W2 IC.

Codex iter-839b recommendation (ae42d036a93fa70f4): before committing
to a broad refactor of the W2 LEGACY production path, compare the
existing ut/vt-based transport (`compute_transport_quantities` +
`transport_step` in `src/legoesm/core/fv_tp_2d.py`) against the
current `fv3_cc2c` + `cgrid_mass_flux_divergence` pair.

Codex iter-841 audit (aae6135c09d2c35ff): `transport_step` takes
CONTRAVARIANT ut, vt + dt and returns h_new (not dh/dt).
`_d_sw1_recompute_ut_vt(uc, vc, cdgrid, dt)` derives contravariant
ut, vt from covariant uc, vc — this is a reusable Fortran-faithful
path (`fv3_sw_core.py:39-89`, used by `_d_sw_native` at 1903-1939).

iter-841 compares dh/dt from the two paths at t=0 on the W2 IC:

  Path A (current production, operators_cdgrid.py:1580-1584):
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)    # grid-aligned cell-centre
    u_c, v_c   = fv3_cc2c(u_cc, v_cc, cdgrid)  # "physical face-normal"
                                                # (asymmetric per iter-839)
    dh_dt_A    = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

  Path B (Fortran-faithful via ut/vt):
    _, _, uc_cov, vc_cov, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)
    ut, vt       = _d_sw1_recompute_ut_vt(uc_cov, vc_cov, cdgrid, dt)
    h_new        = transport_step(h, ut, vt, dt, cdgrid)
    dh_dt_B      = (h_new − h) / dt

Measure |dh_dt_B − dh_dt_A| at W2 IC, report peak location +
distance to nearest cube vertex.  Observational-only — no source-
code change.  Informs iter-842+ decision on whether Path B gives a
materially different (presumably Fortran-faithfuller) signal at
cube vertices.
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

from legoesm.core.fv3_sw_core import _d2a2c_vect, _d_sw1_recompute_ut_vt
from legoesm.core.fv_tp_2d import transport_step
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, cgrid_mass_flux_divergence)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _peak_with_location(arr, cdgrid):
    flat = np.abs(arr).reshape(6, -1)
    face = int(np.argmax(np.max(flat, axis=1)))
    ij = int(np.argmax(flat[face]))
    n = arr.shape[1]
    i, j = ij // n, ij % n
    peak = float(arr[face, i, j])
    lat_c = np.asarray(cdgrid.base.lat)[face, i, j]
    lon_c = np.asarray(cdgrid.base.lon)[face, i, j]
    lat_d, lon_d = float(np.rad2deg(lat_c)), float(np.rad2deg(lon_c))
    if lon_d > 180:
        lon_d -= 360
    CV_LAT = np.arcsin(1.0 / np.sqrt(3.0))
    cv_lats = np.array([CV_LAT, -CV_LAT])
    cv_lons = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))
    min_d = np.inf
    for vlat in cv_lats:
        for vlon in cv_lons:
            dlat = lat_c - vlat
            dlon = lon_c - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_c) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(max(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return peak, face, i, j, lat_d, lon_d, min_d


def main(n=36, dt=300.0):
    # iter-761 canonical config (non-duogrid, matches W2 LEGACY sentinel)
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
    h = sw.h.data

    # --- Path A: current production (fv3_d2cc + fv3_cc2c + cgrid_mass_flux_divergence) ---
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    dh_dt_A = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

    # --- Path B: Fortran-faithful (covariant uc/vc → ut/vt → transport_step) ---
    _, _, uc_cov, vc_cov, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)
    ut, vt = _d_sw1_recompute_ut_vt(uc_cov, vc_cov, cdgrid, dt)
    h_new_B = transport_step(h, ut, vt, dt, cdgrid)
    dh_dt_B = (h_new_B - h) / dt

    dh_dt_A_np = np.asarray(dh_dt_A)
    dh_dt_B_np = np.asarray(dh_dt_B)
    delta = dh_dt_B_np - dh_dt_A_np

    peak_A = float(np.max(np.abs(dh_dt_A_np)))
    peak_B = float(np.max(np.abs(dh_dt_B_np)))
    peak_delta = float(np.max(np.abs(delta)))

    print(f"Iter-841 dh/dt comparison at t=0 on W2 IC, C{n}, dt={dt}s, "
          f"LEGACY (iter-761 canonical)")
    print()
    print(f"Path A (current production: fv3_cc2c + cgrid_mass_flux_divergence):")
    print(f"  peak |dh_dt_A|       = {peak_A:.3e} kg m-2 s-1")
    print(f"Path B (Fortran ut/vt: _d_sw1_recompute_ut_vt + transport_step):")
    print(f"  peak |dh_dt_B|       = {peak_B:.3e} kg m-2 s-1")
    print(f"Delta (B − A):")
    print(f"  peak |delta|         = {peak_delta:.3e} kg m-2 s-1")
    print(f"  relative vs |dh_dt_A|= {peak_delta / peak_A * 100:.2f} %")
    print()

    # Delta peak location
    peak, face, i, j, lat, lon, gc = _peak_with_location(delta, cdgrid)
    print(f"Delta peak at face={face} (i,j)=({i},{j})")
    print(f"  lat = {lat:.2f}°  lon = {lon:.2f}°")
    print(f"  GC to nearest cube vertex = {gc:.2f}°")
    print()

    # For W2 solid-body rotation, the EXACT dh/dt is 0 at every point.
    # So both paths report errors; the one with smaller peak is more
    # Fortran-faithful.
    print("W2 EXACT dh/dt = 0 (solid-body rotation preserves h).")
    print(f"  |dh_dt_A| peak (Path A error) = {peak_A:.3e}")
    print(f"  |dh_dt_B| peak (Path B error) = {peak_B:.3e}")
    if peak_B < peak_A:
        print(f"  → Path B peak is {(1 - peak_B / peak_A) * 100:.1f} % "
              f"SMALLER than Path A.  Candidate Fortran-faithful improvement.")
    elif peak_B > peak_A:
        print(f"  → Path B peak is {(peak_B / peak_A - 1) * 100:.1f} % "
              f"LARGER than Path A.  Path A is closer to W2 steady state.")
    else:
        print(f"  → Paths match within numerical precision.")
    print()

    # Peak location of Path A and Path B errors
    print("Path A error peak location:")
    _, f_a, i_a, j_a, lat_a, lon_a, gc_a = _peak_with_location(
        dh_dt_A_np, cdgrid)
    print(f"  face={f_a} (i,j)=({i_a},{j_a})  lat={lat_a:.2f}°  "
          f"lon={lon_a:.2f}°  GC-to-vertex={gc_a:.2f}°")
    print("Path B error peak location:")
    _, f_b, i_b, j_b, lat_b, lon_b, gc_b = _peak_with_location(
        dh_dt_B_np, cdgrid)
    print(f"  face={f_b} (i,j)=({i_b},{j_b})  lat={lat_b:.2f}°  "
          f"lon={lon_b:.2f}°  GC-to-vertex={gc_b:.2f}°")


if __name__ == "__main__":
    main()
