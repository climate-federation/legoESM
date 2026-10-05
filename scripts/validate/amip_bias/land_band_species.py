#!/usr/bin/env python3
"""Column water by species over a latitude band of LAND cells, from daily checkpoints.

Complements water_phase_partition.py (whole bands, last day only): land-only
mask, a day range, and the window mean, so the radiatively active cloud ice
(q_i) can be set against the radiatively inert snow and graupel.

Land fraction and cell order come from a capture made on the same mesh
(nh_surface_replay.py output: f_land, lat, area).

Usage: land_band_species.py <run_dir> <capture.npz> <day0> <day1> [--lat 45 70]
"""
import argparse
import glob

import numpy as np

from water_phase_partition import SPECIES, column_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("capture")
    ap.add_argument("day0", type=int); ap.add_argument("day1", type=int)
    ap.add_argument("--lat", nargs=2, type=float, default=(45.0, 70.0))
    a = ap.parse_args()
    c = np.load(a.capture, allow_pickle=True)
    lat, fl, area = c["lat"], c["f_land"], c["area"]
    w = area * ((lat >= a.lat[0]) & (lat <= a.lat[1]) & (fl >= 0.5))
    cks = [f for f in sorted(glob.glob(f"{a.run_dir}/checkpoint_day_*.npz"))
           if a.day0 <= int(f[-8:-4]) <= a.day1]
    if not cks:
        raise SystemExit("no checkpoints in range")
    rows = []
    for ck in cks:
        _, p = column_paths(ck)
        if p["trc_q_v"].shape != lat.shape:
            raise SystemExit(f"{ck}: mesh size {p['trc_q_v'].shape} != capture {lat.shape}")
        rows.append([1e3 * float((p[s] * w).sum() / w.sum()) for s in SPECIES])
    m = np.mean(rows, axis=0)
    print(f"{a.run_dir} days {a.day0}-{a.day1} ({len(cks)} checkpoints, 00Z), "
          f"land {a.lat[0]:.0f}-{a.lat[1]:.0f}N, g/m2:")
    print("  " + "  ".join(f"{s[4:]} {v:.1f}" for s, v in zip(SPECIES, m)))


if __name__ == "__main__":
    main()
