#!/usr/bin/env python
"""Audit domain-mean grid metric consistency across grids.

Builds atmosphere matrix (C36, 72x144, ico5, T21) and ocean matrix
(C24, 36x72, ico3) at their canonical resolutions and reports:

  * total_area_rel_err     = (sum(area) - 4 pi R^2) / (4 pi R^2)
  * pair-wise rel_diff     = (A_g - A_ref) / A_ref      across grids
  * mean(area), std(area), area_max/area_min
  * mean(lat_cell), mean(cos(lat)), area-weighted mean(1) = 1.0
  * Coriolis area-weighted mean (must be 0 for sphere)
  * sin(lat) area-weighted mean (must be 0)

Exits non-zero if any consistency check exceeds tolerance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402

R = float(constants.R_earth)
SPHERE_AREA = 4.0 * np.pi * R * R

# Tolerances. Grid metric arrays are stored at the global precision
# policy's storage dtype (float32 by default for FV grids), so total-
# area reductions inherit ~6e-8 of fp32 round-off. Tighter limits
# require an fp64 storage policy.
TOL_TOTAL_AREA = 1.0e-7
TOL_PAIRWISE = 1.0e-7
TOL_SIN_LAT = 1.0e-7


def _build_atmosphere_grids() -> dict:
    """Build the canonical atmosphere-matrix grids."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.gaussian import create_gaussian_grid

    return {
        "cubed_sphere/C36": create_cubed_sphere(36),
        "latlon/72x144": create_latlon_grid(72, 144),
        "icosahedral/ico5": create_voronoi_mesh(5),
        "spectral/T21": create_gaussian_grid(21),
    }


def _build_ocean_grids() -> dict:
    """Build the canonical ocean-matrix grids."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.voronoi import create_voronoi_mesh

    return {
        "cubed_sphere/C24": create_cubed_sphere(24),
        "latlon/36x72": create_latlon_grid(36, 72),
        "mpas/ico3": create_voronoi_mesh(3),
    }


def _metric_row(name: str, grid) -> dict:
    """Compute the canonical metric set for any GridProtocol grid."""
    area = np.asarray(grid.grid_area).ravel()
    lat = np.asarray(grid.grid_lat).ravel()
    cor = np.asarray(grid.grid_coriolis).ravel()
    total = float(area.sum())
    awmean = lambda x: float((x * area).sum() / total)  # noqa: E731
    return dict(
        grid=name,
        n_cells=int(area.size),
        total_area=total,
        rel_err_total=(total - SPHERE_AREA) / SPHERE_AREA,
        mean_area=float(area.mean()),
        std_area=float(area.std()),
        amax_over_amin=float(area.max() / area.min()),
        awmean_sin_lat=awmean(np.sin(lat)),
        awmean_cos_lat=awmean(np.cos(lat)),
        awmean_coriolis=awmean(cor),
        radius=float(grid.grid_radius),
    )


def _report(label: str, grids: dict) -> int:
    print(f"\n===== {label} =====")
    print(f"Reference sphere area: 4πR² = {SPHERE_AREA:.6e} m²  (R={R} m)")
    rows = []
    for name, g in grids.items():
        rows.append(_metric_row(name, g))

    fmt = (
        "{grid:24s}  Ncell={n_cells:>7d}  "
        "tot={total_area:.10e}  drel={rel_err_total:+.2e}  "
        "<area>={mean_area:.3e}  σ/<>={ratio_sigma:.2e}  "
        "max/min={amax_over_amin:.2f}"
    )
    for r in rows:
        r2 = dict(r, ratio_sigma=r["std_area"] / r["mean_area"])
        print(fmt.format(**r2))

    print("Area-weighted moments (should be ~0 for global sphere):")
    for r in rows:
        print(
            f"  {r['grid']:24s}  <sin lat>={r['awmean_sin_lat']:+.3e}  "
            f"<f>={r['awmean_coriolis']:+.3e}  "
            f"<cos lat>={r['awmean_cos_lat']:.6f}  (expect 2/π≈0.6366)"
        )

    # Consistency checks
    fails = 0
    for r in rows:
        if abs(r["rel_err_total"]) > TOL_TOTAL_AREA:
            print(
                f"  FAIL total-area: {r['grid']} drel={r['rel_err_total']:+.3e}"
                f"  > tol {TOL_TOTAL_AREA:.0e}"
            )
            fails += 1
        if abs(r["radius"] - R) > 1e-6:
            print(f"  FAIL radius mismatch: {r['grid']} R={r['radius']} != {R}")
            fails += 1
        if abs(r["awmean_sin_lat"]) > TOL_SIN_LAT:
            print(
                f"  FAIL <sin lat>: {r['grid']} "
                f"{r['awmean_sin_lat']:+.3e} > {TOL_SIN_LAT:.0e}"
            )
            fails += 1
        if abs(r["awmean_coriolis"]) > TOL_SIN_LAT * 2.0 * float(constants.Omega):
            print(
                f"  FAIL <Coriolis>: {r['grid']} "
                f"{r['awmean_coriolis']:+.3e} > tol"
            )
            fails += 1

    # Pairwise total-area cross-grid agreement
    ref_name, ref = rows[0]["grid"], rows[0]["total_area"]
    print(f"\nPairwise total_area diffs (vs {ref_name}):")
    for r in rows[1:]:
        d = (r["total_area"] - ref) / ref
        flag = "" if abs(d) < TOL_PAIRWISE else " FAIL"
        print(f"  {r['grid']:24s}  drel={d:+.3e}{flag}")
        if abs(d) >= TOL_PAIRWISE:
            fails += 1

    print(f"{label}: {fails} fail(s)")
    return fails


def main() -> int:
    fails = 0
    fails += _report("ATMOSPHERE matrix grids", _build_atmosphere_grids())
    fails += _report("OCEAN matrix grids", _build_ocean_grids())
    print(f"\nTotal failures: {fails}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
