#!/usr/bin/env python3
"""Land columns FROZEN for a whole day: top-soil temperature, SWE and top-layer
water all bit-identical between consecutive daily checkpoints.  A land column held
on every land step of a day reverts its state each time, so this counts only
whole-day holds (a lower bound on holding; partial-day holds are invisible).
Active land = columns whose top-soil T changes on at least one day pair.

Usage: nh_frozen_columns.py <run_dir> <mesh.npz> <day0> <day1>
"""
import sys

import numpy as np

REGIONS = {"45-70N": (45, 70, 0, 360), "Siberia": (50, 70, 60, 140),
           "Canada": (50, 70, 220, 300), "Australia": (-40, -10, 110, 155),
           "global": (-90, 90, 0, 360)}
run, mesh, d0, d1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
m = np.load(mesh)
lat, lon = m["lat"], m["lon"] % 360
S = []
for d in range(d0, d1 + 1):
    c = np.load(f"{run}/checkpoint_day_{d:04d}.npz")
    S.append(np.stack([c["land_ml_T_soil"][:, 0], c["land_ml_snow_depth"],
                       c["land_ml_theta_soil"][:, 0]], 1))
S = np.stack(S)                                   # (days, ncol, 3)
same = np.all(S[1:] == S[:-1], axis=2)            # (days-1, ncol)
active = np.any(S[1:, :, 0] != S[:-1, :, 0], axis=0)
for k, (a, b, c_, e) in REGIONS.items():
    r = active & (lat >= a) & (lat <= b) & (lon >= c_) & (lon <= e)
    fz = same[:, r]
    runs = [max((len(s) for s in "".join("1" if v else "0" for v in col).split("0")),
                default=0) for col in fz.T[fz.any(0)]]
    print(f"[{k}] active land columns {int(r.sum())}: whole-day frozen column-days "
          f"{int(fz.sum())} of {fz.size} ({fz.mean():.4f}); columns ever frozen "
          f"{int(fz.any(0).sum())}; longest frozen streak {max(runs) if runs else 0} d "
          f"(days {d0}-{d1})")
