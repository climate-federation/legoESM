#!/usr/bin/env python
"""Southern-Ocean ice-volume tendency from snapshots, on TRUE cell area.

WHY THIS FILE EXISTS.  The first version of this measurement was an inline
heredoc that weighted by ``cos(lat)``.  On the eORCA1 tripole the true cell
area ``e1t*e2t`` differs from a cos(lat) weight by a factor of 0.00 to 1.72
per cell south of 45S (mean 0.771), so that shortcut inflated the reported
rate by roughly 45 percent -- 52.5 instead of 36.4e-6 kg/m2/s at day 5 -- and
a factor-of-3.2 overshoot against NEMO was reported when the truth is 2.2.
Snapshots carry ``lat_T`` and ``land_mask`` but NOT cell area, which is why
the shortcut was tempting; the area comes from the mesh-mask file instead.

WHAT IT MEASURES, named honestly.  The day-to-day change in
``ice_concentration * ice_thickness`` integrated over the band and divided by
the band's ocean area.  That is the ice-VOLUME tendency, not melt: it also
contains ice exported across the band edge, and it EXCLUDES snow (``ht_s``),
whose melt is real freshwater the ocean receives.  Use it as a bound on the
ice freshwater term, and say "volume tendency" when quoting it.

READ AGAINST: NEMO ORCA1's runoff+ice residual south of 45S is 16.67e-6
kg/m2/s (empmr - emp_oce, job 9441497), which bundles runoff with the ice
exchange, so our ice term alone must come in UNDER it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_RHO_ICE_TO_FW = 1000.0 / 86400.0  # m of ice per day -> kg/m2/s of freshwater


def band_ice_volume(snapshot: Path, area: np.ndarray, lat_max: float) -> float:
    """Band ice volume per unit ocean area [m], on true cell area."""
    z = np.load(snapshot)
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    if area.shape != lat.shape:
        raise SystemExit(
            f"mesh area {area.shape} does not match snapshot grid {lat.shape}"
            " -- wrong mesh file for this run")
    wet = (lat < lat_max) & (np.asarray(z["land_mask"]) > 0.5)
    if not wet.any():
        raise SystemExit(f"no wet cells south of {lat_max}")
    w = area[wet]
    vol = (np.asarray(z["ice_concentration"], dtype=np.float64)
           * np.asarray(z["ice_thickness"], dtype=np.float64))[wet]
    return float(np.nansum(vol * w) / np.nansum(w))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--mesh", type=Path,
                    default=Path("data/grids/eORCA1.2_mesh_mask.nc"))
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--lat-max", type=float, default=-45.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import netCDF4 as nc
    d = nc.Dataset(a.mesh)
    try:
        area = (np.squeeze(d["e1t"][:]) * np.squeeze(d["e2t"][:])).astype(np.float64)
        lat_m = np.squeeze(d["gphit"][:]).astype(np.float64)
    finally:
        d.close()

    # Self-check that would have caught the original defect: report how far a
    # cos(lat) weight is from the truth, so the shortcut is never re-adopted
    # silently on a mesh where it happens to look acceptable.
    m = lat_m < a.lat_max
    cw = np.cos(np.deg2rad(lat_m))
    ratio = (area[m] / area[m].sum()) / (cw[m] / cw[m].sum())
    print(f"[weight check] true-area / cos(lat) weight south of {a.lat_max}: "
          f"min {ratio.min():.3f} max {ratio.max():.3f} mean {ratio.mean():.3f}"
          "  (cos(lat) is only valid if all three are ~1)")

    print(f"\n{a.label or a.run_dir.name}: ice-VOLUME tendency, true area, "
          f"1e-6 kg/m2/s (+ = ocean gains freshwater)")
    prev = None
    for i in range(1, a.days + 1):
        snap = a.run_dir / f"snapshot_day{i:04d}.npz"
        if not snap.exists():
            break
        vol = band_ice_volume(snap, area, a.lat_max)
        if prev is not None:
            print(f"   d{i}  {-(vol - prev) * _RHO_ICE_TO_FW * 1e6:7.1f}")
        prev = vol
    if prev is None:
        raise SystemExit(f"no snapshots found in {a.run_dir}")
    print("\nNEMO runoff+ice residual south of 45S: 16.67e-6 (our ice term "
          "alone must be under it, since NEMO's bundles runoff)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
