#!/usr/bin/env python
"""NEMO's OWN per-process temperature budget of an equatorial box, level by level, from
an hourly ``trd1h_T`` file (``ln_tra_trd`` output): which process warms 20-30 m at night?

Reads ``ttrd_zdf`` (vertical diffusion incl. EVD), ``ttrd_zdfp`` (pure vertical diffusion),
``ttrd_evd``, ``ttrd_totad`` (total advection), ``ttrd_qsr`` (penetrative solar),
``ttrd_ldf``/``ttrd_iso`` (lateral/isoneutral), ``ttrd_tot`` (total model trend) and
``wocetr_eff`` (effective vertical transport through the cell TOP face, m3/s, NEMO w
positive UPWARD).  Box means are unweighted over the T-columns in the box (the same
reduction the other equatorial probes use); trends are printed in K/day and ALSO
integrated over the requested record window (sum of hourly means x 3600 s) so the
per-process contribution to the window's temperature CHANGE is read directly in K.

Provenance is stamped; NaN in any read field is fatal (no nanmean).
"""
import argparse
import subprocess
import sys

import netCDF4 as nc
import numpy as np

TERMS = ("ttrd_tot", "ttrd_totad", "ttrd_zdf", "ttrd_zdfp", "ttrd_evd", "ttrd_qsr",
         "ttrd_ldf", "ttrd_iso", "ttrd_bbl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trd-file", required=True, help="ORCA1_1h_*_trd1h_T.nc")
    ap.add_argument("--recs", default="0-47", help="record window lo-hi (inclusive) to integrate")
    ap.add_argument("--print-recs", default="", help="comma list of single records to print as K/day profiles")
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=18)
    ap.add_argument("--mesh-mask", default=None, help="for e1t*e2t (w = wocetr_eff / area); optional")
    a = ap.parse_args()

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha} | {' '.join(sys.argv)}")
    d = nc.Dataset(a.trd_file)
    # XIOS names the T-grid coordinates nav_lat_grid_T when a file mixes T and W grids
    _lat = "nav_lat" if "nav_lat" in d.variables else "nav_lat_grid_T"
    _lon = "nav_lon" if "nav_lon" in d.variables else "nav_lon_grid_T"
    la = np.asarray(d.variables[_lat][:]); lo = np.asarray(d.variables[_lon][:]) % 360.0
    box = (np.abs(la) <= a.lat_halfwidth) & (lo >= a.lon_lo) & (lo <= a.lon_hi)
    dep = np.asarray(d.variables["deptht"][:]).ravel()
    n = a.n_levels
    lo_r, hi_r = (int(x) for x in a.recs.split("-"))
    nrec = d.variables["ttrd_tot"].shape[0]
    hi_r = min(hi_r, nrec - 1)
    print(f"box {a.lon_lo}-{a.lon_hi}E |lat|<={a.lat_halfwidth}: {int(box.sum())} T-columns; records {lo_r}-{hi_r} of {nrec}")

    def box_mean(name, r, k):
        v = np.asarray(d.variables[name][r, k])
        vals = v[box]
        if not np.all(np.isfinite(vals)):
            raise SystemExit(f"{name} rec {r} level {k}: NaN inside the box (fatal)")
        return float(vals.mean())

    have = [t for t in TERMS if t in d.variables]
    # window integral, K
    integ = {t: np.zeros(n) for t in have}
    for r in range(lo_r, hi_r + 1):
        for t in have:
            for k in range(n):
                integ[t][k] += box_mean(t, r, k) * 3600.0
    print(f"\n## window integral rec {lo_r}-{hi_r} ({hi_r - lo_r + 1} h): contribution of each process to dT [K], per level")
    print("   z m  | " + " ".join(f"{t:>10s}" for t in have) + " | residual(tot-sum of parts)")
    parts = [t for t in have if t not in ("ttrd_tot", "ttrd_zdfp", "ttrd_evd")]  # zdf already contains evd; zdfp is the pure part of zdf
    for k in range(n):
        s = sum(integ[t][k] for t in parts)
        res = integ["ttrd_tot"][k] - s if "ttrd_tot" in integ else float("nan")
        print(f"{dep[k]:6.2f} | " + " ".join(f"{integ[t][k]:+10.4f}" for t in have) + f" | {res:+.4f}")

    if "wocetr_eff" in d.variables:
        area = None
        if a.mesh_mask:
            m = nc.Dataset(a.mesh_mask)
            e1 = np.asarray(m.variables["e1t"][0]); e2 = np.asarray(m.variables["e2t"][0])
            # hourly files are the inner domain [1:-1,1:-1] of the 332x362 mesh
            if e1.shape != la.shape:
                e1 = e1[1:1 + la.shape[0], 1:1 + la.shape[1]]; e2 = e2[1:1 + la.shape[0], 1:1 + la.shape[1]]
            area = e1 * e2
        print(f"\n## wocetr_eff box mean, window rec {lo_r}-{hi_r}: m3/s per column" + (" and w = flux/area [m/day], positive UP" if area is not None else ""))
        for k in range(n):
            acc = 0.0; accw = 0.0
            for r in range(lo_r, hi_r + 1):
                v = np.asarray(d.variables["wocetr_eff"][r, k])
                vals = v[box]
                if not np.all(np.isfinite(vals)):
                    raise SystemExit(f"wocetr_eff rec {r} level {k}: NaN inside the box (fatal)")
                acc += vals.mean()
                if area is not None:
                    accw += (vals / area[box]).mean()
            nh = hi_r - lo_r + 1
            print(f"{dep[k]:6.2f} m (top face) | {acc / nh:+.3e} m3/s" + (f" | w {accw / nh * 86400:+.3f} m/day" if area is not None else ""))

    for rs in [x for x in a.print_recs.split(",") if x]:
        r = int(rs)
        print(f"\n## rec {r} (local {(r + 15.3) % 24:04.1f} h at 230E): K/day per level")
        print("   z m  | " + " ".join(f"{t:>10s}" for t in have))
        for k in range(n):
            print(f"{dep[k]:6.2f} | " + " ".join(f"{box_mean(t, r, k) * 86400:+10.4f}" for t in have))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
