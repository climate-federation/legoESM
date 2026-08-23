#!/usr/bin/env python
"""Do the CORE-II base and _MOD wind variables differ in SPEED, or only direction?

WHY THIS IS COMMITTED.  The first version of this measurement was an inline
heredoc, so the adversarial reviewer could not read it and reasonably assumed
the worst: its top finding was that "the alleged 33% wind magnitude is a
component comparison, not a wind-speed comparison", since a rotation can change
the eastward component a great deal while preserving sqrt(U^2+V^2).  That
objection is correct in general and would be fatal if true.  It was not true --
the heredoc already paired U with V through ``hypot`` -- but an uncheckable
probe cannot defend itself.  Hence this file.

WHAT IT DECIDES.  NEMO ORCA1's namelist reads ``U_10_MOD``/``V_10_MOD`` while
our forcing cache is built from ``U_10``/``V_10``.  If the pair differs only by
a rotation, the wind SPEED entering a bulk formula is identical and our cache is
fine.  If the paired speeds differ, we and the oracle are forcing the ocean with
different winds, and the stress difference scales roughly as the square.

It reports, on the file's own T62 grid with no remapping:
  * paired speed for each variant, area-weighted over the box;
  * the mean turning angle between the two vectors, which is what a pure
    rotation would show;
  * the speed ratio, and the stress ratio it implies (speed squared).

CAVEAT, and the reviewer's second finding, which stands: NEMO's ``windsp``
diagnostic is NOT comparable to either number here -- it lives on the ORCA grid
after bicubic remapping and the vector rotation, and is an hourly quantity
averaged over five days.  This probe therefore compares the two INPUT variants
against each other only, which is a self-contained question needing no NEMO
diagnostic at all.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

BOXES = {
    "nino3": (-5.0, 5.0, 210.0, 270.0),
    "nino4": (-5.0, 5.0, 160.0, 210.0),
    "eq_pacific": (-2.0, 2.0, 160.0, 270.0),
    "global": (-90.0, 90.0, 0.0, 360.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core2-dir", default="/burg-archive/glab/users/pg2328/"
                    "nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/")
    ap.add_argument("--rec0", type=int, default=100)
    ap.add_argument("--rec1", type=int, default=120)
    a = ap.parse_args()

    import netCDF4 as nc
    U = nc.Dataset(a.core2_dir + "u_10.15JUNE2009.nc")
    V = nc.Dataset(a.core2_dir + "v_10.15JUNE2009.nc")
    try:
        for need, ds in (("U_10", U), ("U_10_MOD", U),
                         ("V_10", V), ("V_10_MOD", V)):
            if need not in ds.variables:
                raise SystemExit(f"{need} missing from the CORE-II file")
        lat = np.asarray(U["LAT"][:], dtype=np.float64)
        lon = np.asarray(U["LON"][:], dtype=np.float64) % 360.0

        def stack(un, vn):
            u = np.asarray(U[un][a.rec0:a.rec1], dtype=np.float64)
            v = np.asarray(V[vn][a.rec0:a.rec1], dtype=np.float64)
            return u, v

        u_b, v_b = stack("U_10", "V_10")
        u_m, v_m = stack("U_10_MOD", "V_10_MOD")
    finally:
        U.close()
        V.close()

    sp_b = np.hypot(u_b, v_b)          # PAIRED speed, not a component
    sp_m = np.hypot(u_m, v_m)
    # Turning angle between the two vectors, per point and record.
    dot = u_b * u_m + v_b * v_m
    denom = np.maximum(sp_b * sp_m, 1e-12)
    ang = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))

    w2 = np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)[None, :]
    print(f"CORE-II records {a.rec0}-{a.rec1 - 1}, T62 grid "
          f"({lat.size}x{lon.size}), no remapping.\n")
    print(f"  {'box':12s}{'|U| base':>10s}{'|U| _MOD':>10s}{'ratio':>8s}"
          f"{'stress~r^2':>12s}{'turn deg':>10s}")
    for name, (la, lb, oa, ob) in BOXES.items():
        m = ((lat >= la) & (lat <= lb))[:, None] & ((lon >= oa) & (lon <= ob))[None, :]
        if not m.any():
            continue
        w = w2 * m
        sb = float(np.nansum(sp_b.mean(axis=0) * w) / np.nansum(w))
        sm = float(np.nansum(sp_m.mean(axis=0) * w) / np.nansum(w))
        an = float(np.nansum(ang.mean(axis=0) * w) / np.nansum(w))
        print(f"  {name:12s}{sb:10.3f}{sm:10.3f}{sm / sb:8.3f}"
              f"{(sm / sb) ** 2:12.3f}{an:10.2f}")
    print("\nA PURE ROTATION would give ratio 1.000 with a non-zero turning "
          "angle.\nA ratio away from 1 means the two variants carry different "
          "wind SPEED,\nso a bulk formula fed one or the other sees a different "
          "stress.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
