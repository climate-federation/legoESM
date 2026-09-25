#!/usr/bin/env python
"""Equatorial SST bias against NEMO's five-day mean, per snapshot CLOCK PHASE.

Every equatorial SST comparison in this campaign pairs OUR 00:00 UTC snapshot
(13:20-18:00 local at 200-270E, the afternoon of the --dm2dc diurnal warm
layer) with NEMO's FIVE-DAY MEAN. This prints the same box biases for
snapshots taken at several UTC phases of one day, through the scorer's own
loaders and regrid (compare_omip_nemo), so the phase dependence of the
"warm lens" is measured rather than assumed. Also prints our own SST minus
our ~10 m temperature per phase: the diurnal warm layer's amplitude.

The three-way scorer refuses sub-daily snapshots by design (its oracle-record
gate); this probe exists only for the phase question and takes no verdict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))  # scripts/validate
from compare_omip_nemo import _load_legoesm, _load_nemo, regrid_curv_to_latlon  # noqa: E402

BOXES = (("nino3", -5.0, 5.0, 210.0, 270.0),
         ("nino34", -5.0, 5.0, 190.0, 240.0),
         ("eq_pacific", -2.0, 2.0, 180.0, 280.0))


def _box(tgt_lat, tgt_lon, lat0, lat1, lon0, lon1):
    la = (tgt_lat >= lat0) & (tgt_lat <= lat1)
    lo = (tgt_lon >= lon0) & (tgt_lon <= lon1)
    return la[:, None] & lo[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--res-deg", type=float, default=1.0)
    ap.add_argument("--sub-depth-m", type=float, default=10.0)
    a = ap.parse_args()
    tgt_lat = -90.0 + a.res_deg / 2 + a.res_deg * np.arange(int(180 / a.res_deg))
    tgt_lon = a.res_deg / 2 + a.res_deg * np.arange(int(360 / a.res_deg))
    N = _load_nemo(a.nemo_gridt, a.rec)
    sstN, ocN = regrid_curv_to_latlon(N["sst"], N["lat"], N["lon"], N["mask"], tgt_lat, tgt_lon)
    area = np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :]
    print(f"NEMO {Path(a.nemo_gridt).name} record {a.rec} (5-day mean); target {a.res_deg} deg")
    hdr = f"{'snapshot':28s} {'day':>6s} " + "".join(f"{b[0]+' bias':>14s}" for b in BOXES) \
        + "".join(f"{b[0]+' SST-T'+str(int(a.sub_depth_m)):>16s}" for b in BOXES) + f"{'n nino3':>9s}"
    print(hdr)
    for p in a.snapshots:
        L = _load_legoesm(p)
        s = np.load(p)
        z = np.asarray(s["z_center_ref"])
        k = int(np.argmin(np.abs(z - a.sub_depth_m)))
        Tsub = np.asarray(s["T"])[..., k]
        day = float(s["time_days"]) if "time_days" in s.files else float("nan")
        sstL, ocL = regrid_curv_to_latlon(L["sst"], L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
        subL, _ = regrid_curv_to_latlon(Tsub, L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
        ok = (ocL > 0.5) & (ocN > 0.5) & np.isfinite(sstL) & np.isfinite(sstN)
        row = f"{Path(p).name:28s} {day:6.2f} "
        biases, dl, n3 = [], [], 0
        for name, la0, la1, lo0, lo1 in BOXES:
            m = ok & _box(tgt_lat, tgt_lon, la0, la1, lo0, lo1)
            w = area[m]
            biases.append(float(np.sum(w * (sstL[m] - sstN[m])) / np.sum(w)))
            dl.append(float(np.sum(w * (sstL[m] - subL[m])) / np.sum(w)))
            if name == "nino3":
                n3 = int(m.sum())
        row += "".join(f"{b:+14.3f}" for b in biases) + "".join(f"{d:+16.3f}" for d in dl) + f"{n3:9d}"
        print(row)
    print(f"(z level used for the sub-surface temperature: {z[k]:.2f} m)")


if __name__ == "__main__":
    main()
