"""Iter-848 diagnostic: div_damp sweep on W2 LEGACY.

iter-846 observed that the W2 LEGACY dv/dt at t=0 has two distinct
peaks: a pure-dynamics peak (1.15e−05, face 5 (0,0), GC=1.16° from
cube vertex) and a full-term peak (1.90e−05, face 0 (2,34), GC=4.34°).
At the pure-dynamics peak, div_damp + boundary_fix partially CANCEL
the dynamics residual; at the full peak, damping DOMINATES.

iter-846 speculated (since retracted in iter-847) that "uniform
damping tuning cannot reduce both peaks at once."  Iter-848 tests
this sign-structure hypothesis directly by scaling div_damp across
{0, 0.25, 0.5, 1.0, 2.0}× the iter-761 canonical and measuring:

  - full-term peak |dv/dt| (over the whole domain)
  - full-term peak LOCATION (face, i, j, lat, lon, GC-to-vertex)
  - |dv/dt| at the iter-846 full-peak location  (face 0 (2,34))
  - |dv/dt| at the iter-846 pure-dynamics peak  (face 5 (0,0))

If reducing div_damp lowers the (2,34) signal but grows the (0,0)
signal on the tested 0..2× range, iter-846's reasoning is consistent
with the data.  If both move in the same direction, the prediction is
NOT SUPPORTED on the tested range — though five samples do not exclude
a narrow non-monotone sub-interval, and this t=0 measurement does NOT
predict the 1-day v_ll stripe behaviour (iter-794's 24h sweep at the
same div_damp scales WORSENS v_ll_Linf when div_damp is reduced).

Observational only.  No source-code change.
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
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_to_cube_vertex_deg(lat_rad, lon_rad):
    best = np.inf
    for vlat in CUBE_VERTEX_LATS:
        for vlon in CUBE_VERTEX_LONS:
            dlat = lat_rad - vlat
            dlon = lon_rad - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_rad) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(max(a, 0.0)))
            best = min(best, np.rad2deg(d))
    return float(best)


def _peak_info(dv, lat, lon):
    arr = np.asarray(dv)
    flat = np.abs(arr).reshape(6, -1)
    face = int(np.argmax(np.max(flat, axis=1)))
    ij = int(np.argmax(flat[face]))
    n_p1 = arr.shape[2]
    i, j = ij // n_p1, ij % n_p1
    lat_c = float(np.asarray(lat)[face, i, j])
    lon_c = float(np.asarray(lon)[face, i, j])
    return {
        "face": face, "i": i, "j": j,
        "peak_abs": float(np.abs(arr[face, i, j])),
        "value": float(arr[face, i, j]),
        "lat_deg": float(np.rad2deg(lat_c)),
        "lon_deg": float(np.rad2deg(lon_c)),
        "gc_deg": _gc_to_cube_vertex_deg(lat_c, lon_c),
    }


def main():
    n = 36
    div_damp_base = 1.5e7 * (48.0 / n) ** 2
    div_damp_canonical = 8.0 * div_damp_base  # iter-761 canonical

    # iter-846 baseline peak indices
    full_peak_idx = (0, 2, 34)
    pure_peak_idx = (5, 0, 0)

    print(f"Iter-848 div_damp sweep on W2 LEGACY C{n}, iter-761 config")
    print(f"Canonical div_damp = {div_damp_canonical:.3e}")
    print()
    print(f"{'scale':>6}  {'peak |dv/dt|':>14}  {'peak face (i,j)':>16}  "
          f"{'GC°':>6}  {'|dv/dt| @(0,2,34)':>18}  {'|dv/dt| @(5,0,0)':>17}")
    print("-" * 96)

    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

    for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
        div_damp = scale * div_damp_canonical
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=div_damp,
            boundary_fix=True,
            damp_v=0.06,
            nord_v=2,
        )
        model = FV3EdgeShallowWaterModel(grid, config=cfg)
        cdgrid = model.cdgrid
        u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
        v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

        # Compute dv/dt at t=0 via the production tendency function.
        # fv3_sw_tendencies reads config via closure; re-obtain dh/du/dv directly.
        _, _, dv_d = fv3_sw_tendencies(
            sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
            g=9.80616,
            div_damp=div_damp,
            hyperdiff_coeff=0.0,
            boundary_fix=True,
            boundary_fix_skip_corners=False,
            fortran_dir_aware_corners=False,
            fortran_a2b_corner_avg=False,
            fortran_vector_corner_fill=False,
            zero_mean_correction=False,
        )

        peak = _peak_info(dv_d, cdgrid.lat_edge_y, cdgrid.lon_edge_y)
        val_at_full = float(np.asarray(dv_d)[full_peak_idx])
        val_at_pure = float(np.asarray(dv_d)[pure_peak_idx])

        print(f"{scale:>6.2f}  {peak['peak_abs']:>14.6e}  "
              f"{peak['face']} ({peak['i']:>2},{peak['j']:>2})    "
              f"{peak['gc_deg']:>5.2f}  {abs(val_at_full):>18.6e}  "
              f"{abs(val_at_pure):>17.6e}")

    print()
    print("Interpretation (observational only):")
    print("- iter-846 sign-structure prediction was that |dv/dt| @(0,2,34)")
    print("  and |dv/dt| @(5,0,0) move in OPPOSITE directions with damping")
    print("  scale.  If they move in opposite directions on the tested range,")
    print("  iter-846's reasoning is consistent with the data; if they move")
    print("  in the same direction, the prediction is NOT SUPPORTED on the")
    print("  tested range (does not exclude narrow non-monotone sub-intervals).")
    print("- Peak LOCATION shifts with scale → damping is influencing the")
    print("  full-peak position.")
    print("- IMPORTANT: this t=0 single-step measurement does NOT predict the")
    print("  1-day v_ll stripe behaviour.  iter-794's 24h sweep at the same")
    print("  div_damp scales found that lowering div_damp WORSENS 1-day")
    print("  v_ll_Linf (0.159 → 0.234 m/s, +47%).  Do not extrapolate from")
    print("  this t=0 reading to the late-time mode-A mechanism.")


if __name__ == "__main__":
    main()
