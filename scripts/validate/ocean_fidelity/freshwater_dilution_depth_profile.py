"""Where does surface freshwater end up: surface cell or whole column?

Under the virtual-salt closure the freshwater dilution is a TOP-CELL salt
tendency; under ``real_freshwater`` it enters as volume and the z* stretch
conserves h*S per level.  If the stretch is UNIFORM over the column and the
vertical velocity does not carry the surface volume flux, the dilution is
spread over the full depth (a 1 m signal smeared over 2000 m) -- which would
explain, with one mechanism, the too-salty Amazon plume, the Antarctic summer
surface too salty and the Arctic winter surface too fresh under real-FW.

Reads two snapshots (same grid, same IC, same day) plus the IC-equivalent
(day-0 or day-15) and prints, for a lat/lon box, the mean salinity anomaly
S(day) - S(ref) per level: surface-trapped vs depth-uniform.
"""
from __future__ import annotations

import argparse

import numpy as np

BOXES = {"Amazon 0-5N 305-315E": (0, 5, 305, 315),
         "Weddell 70-62S 320-360E": (-70, -62, 320, 360),
         "Kara/Laptev 72-78N 70-140E": (72, 78, 70, 140)}


def load(path):
    z = np.load(path)
    lat = np.asarray(z["lat_T"], float); lon = np.asarray(z["lon_T"], float) % 360
    if np.abs(lat).max() < 7:
        lat, lon = np.degrees(lat), np.degrees(lon) % 360
    return lat.ravel(), lon.ravel(), np.asarray(z["S"]).reshape(lat.size, -1), \
        (np.asarray(z["land_mask"]).ravel() > 0.5), np.abs(np.asarray(z["z_center_ref"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference snapshot (day 0 / earliest)")
    p.add_argument("--arm", action="append", required=True, help="label=snapshot")
    a = p.parse_args()
    lat, lon, S0, wet, zc = load(a.ref)
    arms = []
    for spec in a.arm:
        lab, path = spec.split("=", 1)
        _, _, S, _, _ = load(path); arms.append((lab, S))
    levels = [0, 1, 2, 4, 6, 9, 14, 19, 24, 29, 34, 39, 49, 59]
    for name, (la0, la1, lo0, lo1) in BOXES.items():
        m = wet & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        print(f"\n{name}: {int(m.sum())} wet cells; mean S anomaly vs ref [PSU] per level")
        print("  level  depth   " + "  ".join(f"{lab:>12s}" for lab, _ in arms))
        for k in levels:
            if k >= S0.shape[1]:
                break
            row = []
            for lab, S in arms:
                d = S[m, k] - S0[m, k]
                d = d[np.isfinite(d) & (S0[m, k] > 0.5)]
                row.append(f"{d.mean():+12.4f}" if d.size else f"{'n/a':>12s}")
            print(f"  {k:5d} {zc[k]:6.0f}   " + "  ".join(row))


if __name__ == "__main__":
    main()
