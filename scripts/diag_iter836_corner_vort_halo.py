"""Iter-836 diagnostic: `_corner_vorticity` halo-strategy fidelity.

Codex iter-836 fidelity check (agentId a56dfc70cb22ec09d) identified
`src/legoesm/core/fv3_sw_core.py::_corner_vorticity:1177-1178` as not
Fortran-faithful: the duogrid branch uses `jnp.pad(..., mode='edge')`
on `fx_circ = uc*dxc` and `fy_circ = vc*dyc`, whereas Fortran relies
on the cross-face halo-exchanged uc/vc (from `d2a2c_vect` applied to
halo-filled u/v via `ext_vector`).

iter-836 quantifies the halo-strategy error:

  1. Build W2 IC at C24 with duogrid.
  2. Run `_d2a2c_vect` (which dispatches to the duogrid variant) to
     get interior uc/vc.
  3. Compute vort_abs two ways:
     (a) current code: mode='edge' padding on fx_circ, fy_circ.
     (b) proposed fix: halo-exchange uc/vc via cell-centre averaging
         + `pad_halo_vector` (cross-face rotation-aware), then
         reconstruct fx_circ_pad, fy_circ_pad at halo positions.
  4. Report |vort_abs_edge - vort_abs_rotated| at all corners, with
     the peak location and its distance to the nearest cube vertex.

If the peak is small (≪ 1 % of the interior vort_abs scale), the
halo-strategy error is cosmetic.  If large, it's a genuine
Fortran-fidelity bug and motivates the halo fix.
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

from legoesm.core.fv3_sw_core import _d2a2c_vect, _corner_vorticity
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_vector
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _vort_abs_edge_mode(uc, vc, cdgrid):
    """Reproduce the PRE-iter-836 mode='edge' padding path so the
    diagnostic can measure the halo-strategy error without relying on
    the current `_corner_vorticity` implementation.  Interior
    fx_circ/fy_circ are preserved EXACTLY; halo rows use plain
    mode='edge' copy (same-face extension, no cross-face rotation).
    """
    fx_circ = uc * cdgrid.dxc   # (6, n+1, n)
    fy_circ = vc * cdgrid.dyc   # (6, n, n+1)
    fx_pad = jnp.pad(fx_circ, [(0, 0), (0, 0), (1, 1)], mode='edge')
    fy_pad = jnp.pad(fy_circ, [(0, 0), (1, 1), (0, 0)], mode='edge')
    vort = (fx_pad[:, :, :-1] - fx_pad[:, :, 1:]
            - fy_pad[:, :-1, :] + fy_pad[:, 1:, :])
    rarea_c = 1.0 / cdgrid.area_corner
    return cdgrid.f_corner + rarea_c * vort


def _find_peak_with_location(arr, cdgrid):
    """Return (peak, face, i, j, lat_deg, lon_deg, gc_to_cube_vertex_deg)."""
    flat = np.abs(arr).reshape(6, -1)
    face = int(np.argmax(np.max(flat, axis=1)))
    ij = int(np.argmax(flat[face]))
    n_plus_1 = arr.shape[2]
    i = ij // n_plus_1
    j = ij % n_plus_1
    peak = float(arr[face, i, j])

    lat_c = np.asarray(cdgrid.lat_corner)[face, i, j]
    lon_c = np.asarray(cdgrid.lon_corner)[face, i, j]
    lat_deg = float(np.rad2deg(lat_c))
    lon_deg = float(np.rad2deg(lon_c))
    if lon_deg > 180:
        lon_deg -= 360

    # Cube-vertex positions (rad)
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
    return peak, face, i, j, lat_deg, lon_deg, min_d


def main(n=24):
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
        damp_v=0.0, nord_v=0, d4_bg=0.16, nord=1, fix_mass=False)
    model = FV3FBShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    # Get uc, vc from d2a2c_vect (duogrid path)
    ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)

    # PRE-iter-836 mode='edge' path (reconstructed locally)
    vort_abs_edge = _vort_abs_edge_mode(uc, vc, cdgrid)
    # POST-iter-836b fix: current _corner_vorticity with use_duogrid=True
    vort_abs_rot = _corner_vorticity(uc, vc, cdgrid, use_duogrid=True)

    vort_abs_edge_np = np.asarray(vort_abs_edge)
    vort_abs_rot_np = np.asarray(vort_abs_rot)
    delta = vort_abs_rot_np - vort_abs_edge_np

    print(f"Iter-836 `_corner_vorticity` halo-strategy error at C{n} "
          f"on W2 IC (duogrid)")
    print()
    print("vort_abs INTERIOR scale:")
    peak_int = np.max(np.abs(vort_abs_edge_np))
    print(f"  max |vort_abs_edge|   = {peak_int:.3e} /s")
    print()
    print("Halo-strategy delta (vort_abs_rot - vort_abs_edge):")
    print(f"  peak |delta|          = {float(np.max(np.abs(delta))):.3e} /s")
    print(f"  relative peak         = "
          f"{float(np.max(np.abs(delta))) / peak_int * 100:.3f} % "
          f"of interior scale")
    print()

    # Peak location
    peak, face, i, j, lat, lon, gc = _find_peak_with_location(delta, cdgrid)
    print(f"Peak delta at face={face} (i,j)=({i},{j})")
    print(f"  lat = {lat:.2f}°  lon = {lon:.2f}°")
    print(f"  GC to nearest cube vertex = {gc:.2f}°")
    print()

    # Restrict to PANEL-EDGE corners only (i ∈ {0, n} or j ∈ {0, n})
    mask = np.zeros_like(delta, dtype=bool)
    mask[:, 0, :] = True
    mask[:, n, :] = True
    mask[:, :, 0] = True
    mask[:, :, n] = True
    panel_peak = float(np.max(np.abs(delta[mask])))
    interior_mask = ~mask
    interior_peak = float(np.max(np.abs(delta[interior_mask])))
    print(f"Delta at PANEL-EDGE corners (i∈{{0,n}} or j∈{{0,n}}):"
          f"  peak {panel_peak:.3e} /s")
    print(f"Delta at INTERIOR corners:  peak {interior_peak:.3e} /s")
    print()
    print("Interpretation cues:")
    print("- If panel_peak >> interior_peak, the halo strategy matters")
    print("  only at panel-edge corners (expected).")
    print("- If relative panel_peak >> 1 %, the mode='edge' padding is a")
    print("  real Fortran-fidelity bug; motivates the halo fix.")
    print("- If < 0.1 %, the fix is cosmetic.  Mode='edge' at this halo")
    print("  layer is numerically close to the cross-face value for")
    print("  solid-body rotation (smooth field).")


if __name__ == "__main__":
    main()
