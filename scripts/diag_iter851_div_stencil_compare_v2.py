"""Iter-851 diagnostic (apples-to-apples-ish replacement for iter-850).

iter-850 was RETRACTED because its "production divergence at corners"
was a fabricated 4-point `mode='edge'` average to corners — not part
of the production pipeline.  Production only uses `cgrid_divergence`
at CELL CENTRES; the corner divergence quantity does not exist in the
A-L tendency path.

iter-851 reduces the Fortran corner-divergence to cell centres via
4-point averaging and compares to the production cell-centre
`cgrid_divergence`.  Both operands are now at cell-centre stagger,
which is closer to apples-to-apples than iter-850's invalid
fabricated corner field.

CAVEATS (iter-852 honesty pass, ANY conclusions from this diag must
acknowledge these):
1. The "Fortran-cc" field is ALSO a constructed quantity.  Fortran
   d_sw5 adds damp·delpc to `ke` at CORNER positions, then d_sw6
   takes corner-to-corner gradients of `ke` to update u/v.  Fortran
   NEVER evaluates a cell-centre delpc.  4-point centre-from-corner
   average is one possible reduction, not what Fortran does.
2. 4-point centre averaging blurs by ~1 cell (O(dx²) stencil error).
   The cube-vertex-adjacent peak location may reflect blurring, not
   a pure stencil signature.
3. The `_d_sw5_corner_divergence` nord=0 branch still uses
   `mode='edge'` halo padding for vort/ptc at fv3_sw_core.py:~1044
   (same-face halo gap, analogous to iter-836's _corner_vorticity
   issue).  Fortran-faithful CGRID halo would require fixing that
   first.

The reported 73×–403× same-location ratios therefore MIX:
  (a) genuine Fortran-vs-Python stencil-construction differences,
  (b) the Fortran helper's internal mode='edge' artefact,
  (c) the 4-point centre-from-corner reduction artefact.
This diagnostic cannot disentangle (a), (b), (c).  The honest claim
is "Fortran corner-stencil + 4-pt-avg-to-cc gives substantially
different values from production cgrid_divergence at cube-vertex-
adjacent cells, with mixed contributions from these three sources."

Method:
  1. Production: `div_centre_prod = cgrid_divergence(u_c, v_c, cdgrid)`
     — flux-form cell-centre divergence (the actual production field).
  2. Fortran-via-corner: `delpc_corner = _d_sw5_corner_divergence(...) /
     da_min_c` (extract raw delpc), then 4-point-average to cell
     centres: `delpc_cc[i,j] = 0.25*(delpc_corner[i, j] +
     delpc_corner[i+1, j] + delpc_corner[i, j+1] +
     delpc_corner[i+1, j+1])`.  This is the closest match production
     could USE — though it's still NOT what production currently does
     (production only uses div at cell centres directly).

  3. Compare div_centre_prod vs delpc_cc at cell-centre stagger.
  4. Report peak |Δ|, peak location, GC-to-cube-vertex.

Caveats (iter-851):
- The `_d_sw5_corner_divergence` nord=0 branch still uses
  `mode='edge'` halo padding for vort/ptc at fv3_sw_core.py:~1044.
  Fortran-faithful comparison would require fixing that halo first.
- 4-point centre averaging of corner field introduces O(dx²) spatial
  blurring; comparison is approximate, not bit-exact.

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

    # ----- Production cell-centre divergence (the ACTUAL production field) -----
    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    div_centre_prod = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n)

    # ----- Fortran corner delpc → cell centre via 4-point average -----
    ua, va, _, _, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)
    da_min_c = float(jnp.min(cdgrid.area_corner))
    ke_damping = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdgrid, dt,
        d2_bg=1.0, dddmp=0.0, nord=0,
    )
    delpc_corner = np.asarray(ke_damping) / da_min_c  # (6, n+1, n+1)
    # 4-point average from corners (n+1, n+1) to centres (n, n)
    delpc_cc = 0.25 * (
        delpc_corner[:, :-1, :-1] + delpc_corner[:, 1:, :-1]
        + delpc_corner[:, :-1, 1:] + delpc_corner[:, 1:, 1:])  # (6, n, n)

    # ----- Compare at cell-centre stagger (apples-to-apples) -----
    lat_centre = cdgrid.base.lat
    lon_centre = cdgrid.base.lon

    prod_peak = _peak(div_centre_prod, lat_centre, lon_centre)
    fort_peak = _peak(delpc_cc, lat_centre, lon_centre)

    print(f"Iter-851 cell-centre divergence stencil comparison "
          f"(apples-to-apples-ish; see iter-852 caveats)")
    print(f"W2 IC, C{n}, LEGACY, t=0")
    print()
    print("Production cgrid_divergence (cell-centre flux-form):")
    print(f"  peak |div| = {prod_peak['peak']:.3e} 1/s")
    print(f"  at face={prod_peak['face']} (i,j)=({prod_peak['i']},"
          f"{prod_peak['j']}), lat={prod_peak['lat_deg']:.2f}°, "
          f"lon={prod_peak['lon_deg']:.2f}°, GC={prod_peak['gc_deg']:.2f}°")
    print()
    print("Fortran _d_sw5_corner_divergence delpc → 4-pt avg to cell centre:")
    print(f"  peak |div| = {fort_peak['peak']:.3e} 1/s")
    print(f"  at face={fort_peak['face']} (i,j)=({fort_peak['i']},"
          f"{fort_peak['j']}), lat={fort_peak['lat_deg']:.2f}°, "
          f"lon={fort_peak['lon_deg']:.2f}°, GC={fort_peak['gc_deg']:.2f}°")
    print()

    # Same-location ratios at the two peaks.
    f_p, i_p, j_p = prod_peak["face"], prod_peak["i"], prod_peak["j"]
    prod_at_prod = float(np.asarray(div_centre_prod)[f_p, i_p, j_p])
    fort_at_prod = float(delpc_cc[f_p, i_p, j_p])
    print("Same-location ratios at cell centres (Fortran/production):")
    if abs(prod_at_prod) > 0:
        ratio = fort_at_prod / prod_at_prod
        print(f"  AT production peak (face {f_p} ({i_p},{j_p}), GC={prod_peak['gc_deg']:.2f}°):")
        print(f"    production = {prod_at_prod:.3e}, "
              f"Fortran-cc = {fort_at_prod:.3e}, ratio = {ratio:.3f}")
    f_f, i_f, j_f = fort_peak["face"], fort_peak["i"], fort_peak["j"]
    prod_at_fort = float(np.asarray(div_centre_prod)[f_f, i_f, j_f])
    fort_at_fort = float(delpc_cc[f_f, i_f, j_f])
    if abs(prod_at_fort) > 0:
        ratio2 = fort_at_fort / prod_at_fort
        print(f"  AT Fortran-cc peak (face {f_f} ({i_f},{j_f}), GC={fort_peak['gc_deg']:.2f}°):")
        print(f"    production = {prod_at_fort:.3e}, "
              f"Fortran-cc = {fort_at_fort:.3e}, ratio = {ratio2:.3f}")
    print()
    print("Caveats (iter-852 honesty pass):")
    print("- Fortran-cc is a CONSTRUCTED quantity.  Fortran d_sw5 adds")
    print("  damp·delpc to ke at corner positions; d_sw6 then takes corner-")
    print("  to-corner gradients of ke.  Fortran NEVER evaluates a cell-")
    print("  centre delpc.  4-pt centre-from-corner average is one possible")
    print("  reduction, NOT what Fortran does.")
    print("- Fortran helper's nord=0 branch still uses mode='edge' halo")
    print("  padding for vort/ptc (fv3_sw_core.py:~1044) — analogous to")
    print("  iter-836's _corner_vorticity halo gap.  Halo-fix would refine")
    print("  the comparison.")
    print("- 4-point centre-from-corner averaging introduces O(dx²)")
    print("  spatial blurring; comparison is approximate, not bit-exact.")
    print("- The 73×–403× ratios MIX (a) genuine stencil-construction")
    print("  differences, (b) the helper's mode='edge' halo artefact,")
    print("  (c) the 4-pt centre-from-corner reduction artefact.  This")
    print("  diagnostic CANNOT disentangle (a), (b), (c).  The honest")
    print("  claim is that the two stencil pipelines give substantially")
    print("  different cell-centre values at cube-vertex-adjacent cells,")
    print("  with mixed contributions from these three sources.")


if __name__ == "__main__":
    main()
