#!/usr/bin/env python
"""Hour-by-hour equatorial T/S profile tendency: ours vs NEMO's own hourly rerun.

The turbulent layer in the 220-240E box is 19 m in ours and 36 m in NEMO's
five-day mean, and the hourly NEMO rerun shows both models START at 18.5 m:
NEMO deepens 9-13 m every night and keeps 23-27 m by day; ours deepens 4 m
and collapses to 14 m by day. By hour 5 our N2 at 9-13 m is already 2-3x
NEMO's. This prints the box-mean T and S profiles of both models at matched
UTC hours, and each model's change since NEMO's first hour, so the first
hours where the two profiles part company can be read level by level.

NEMO record h is the mean over hour h..h+1; our snapshot is instantaneous, so
pair rec 5 with our 06 UTC snapshot (half an hour apart, noted in the header).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equatorial_shear_ri_vs_nemo import _box_mean  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo-dir", required=True, type=Path)
    ap.add_argument("--nemo-stem", default="ORCA1_1h_20000101_20000102")
    ap.add_argument("--nemo-suffix", default="_1h")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="rec:snapshot.npz pairs, e.g. 5:/path/snapshot_day0000.250.npz")
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=16)
    a = ap.parse_args()
    nl = a.n_levels

    d = nc.Dataset(a.nemo_dir / f"{a.nemo_stem}_grid_T{a.nemo_suffix}.nc")
    la = np.asarray(d.variables["nav_lat"][:], float); lo = np.asarray(d.variables["nav_lon"][:], float) % 360.0
    dep = np.asarray(d.variables["deptht"][:nl], float)
    boxN = ((np.abs(la) <= a.lat_halfwidth) & (lo >= a.lon_lo) & (lo < a.lon_hi)).ravel()
    e1 = e2 = None
    wN = np.ones(boxN.size)

    def nemo_prof(var, rec):
        x = np.ma.filled(np.ma.masked_invalid(d.variables[var][rec, :nl]), np.nan).astype(float)  # (nl, y, x)
        x = np.where(np.abs(x) < 1e-6, np.nan, x)
        return _box_mean(x.reshape(nl, -1).T, wN, boxN)

    T0, S0 = nemo_prof("to", 0), nemo_prof("so", 0)
    print(f"box {a.lon_lo}-{a.lon_hi}E |lat|<={a.lat_halfwidth}: NEMO {int(boxN.sum())} T-cols (unweighted box mean of the hourly means); "
          f"ours area-weighted instantaneous. Reference = NEMO rec 0 (hour 0-1 mean). dT in K, dS in g/kg (x1e3 shown for S).")
    for pair in a.pairs:
        rec, snap = pair.split(":", 1); rec = int(rec)
        s = np.load(snap)
        T = np.asarray(s["T"], float); S = np.asarray(s["S"], float)
        lat, lon = np.asarray(s["lat_T"], float), np.asarray(s["lon_T"], float) % 360.0
        wet = np.asarray(s["land_mask"], float) > 0.5
        zc = np.abs(np.asarray(s["z_center_ref"], float))[:nl]
        area = np.asarray(s["cell_area"], float).ravel()
        box = (wet & (np.abs(lat) <= a.lat_halfwidth) & (lon >= a.lon_lo) & (lon < a.lon_hi)).ravel()
        Tf = np.where(np.abs(T) < 1e-6, np.nan, T).reshape(-1, T.shape[-1])[:, :nl]
        Sf = np.where(np.abs(S) < 1e-6, np.nan, S).reshape(-1, S.shape[-1])[:, :nl]
        To, So = _box_mean(Tf, area, box), _box_mean(Sf, area, box)
        TN, SN = nemo_prof("to", rec), nemo_prof("so", rec)
        # Our T points sit at mid-cell (e3t/2 = 0.512 m); NEMO's gdept_1d is the
        # analytic value (0.506 m). A 1-2 cm offset in the top 30 m; tolerated,
        # printed once, and a harmonization row on its own.
        if np.max(np.abs(zc - dep)) > 0.6:  # 1-7 cm per level in the top 30 m, up to ~0.5 m by 80 m (mid-cell vs gdept_1d)
            raise SystemExit(f"vertical grids differ: ours {zc[:4]} vs NEMO {dep[:4]}")
        print(f"  (T-level depth offset ours-NEMO, top {nl}: max {np.max(np.abs(zc - dep))*100:.1f} cm)")
        print(f"\n##### NEMO rec {rec} (UTC {rec:02d}-{rec+1:02d}h mean) vs ours {Path(snap).name} (t = {float(s['time_days'])*24:.1f} h); "
              f"{int(box.sum())} our cols")
        print("   z m |   T NEMO   T ours   T o-N |  dT_N(t-0)  dT_o(t-0)  ddT | S NEMO  S ours | dS_N e3 dS_o e3")
        for k in range(nl):
            print(f"{dep[k]:6.2f} | {TN[k]:8.3f} {To[k]:8.3f} {To[k]-TN[k]:7.3f} | "
                  f"{TN[k]-T0[k]:9.3f} {To[k]-T0[k]:9.3f} {(To[k]-T0[k])-(TN[k]-T0[k]):6.3f} | "
                  f"{SN[k]:6.3f} {So[k]:6.3f} | {1e3*(SN[k]-S0[k]):7.1f} {1e3*(So[k]-S0[k]):7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
