"""Vertical advective CFL of the 1 m NEMO top cell at a candidate time step.

GLM review 2026-09-05: before running FESOM on NEMO's 75-level ladder at
FESOM's dt = 1800 s, measure w*dt/dz_top from an existing baseline that already
runs that ladder (tripole, dt 150) -- the vertical velocity is a property of the
flow, the CFL number of the time step.  Reports the distribution of
|w_1| * dt / dz_top for the requested dt values, the max and where it sits.
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dt", type=float, nargs="+", default=[150.0, 1800.0, 3600.0])
    a = p.parse_args()
    z = np.load(a.snapshot)
    w = np.asarray(z["mass_flux_w"], dtype=np.float64)        # (..., nlev+1) interfaces
    wet = np.asarray(z["land_mask"]) > 0.5
    zc = np.abs(np.asarray(z["z_center_ref"]))
    dz_top = 2.0 * zc[0]
    lat = np.asarray(z["lat_T"], float); lon = np.asarray(z["lon_T"], float)
    if np.abs(lat).max() < 7:
        lat, lon = np.degrees(lat), np.degrees(lon)
    w1 = np.abs(w[..., 1])[wet]                                # first interior interface
    print(f"{a.snapshot}: dz_top {dz_top:.3f} m, wet cells {w1.size}, |w_1| p50/p99/max = "
          f"{np.percentile(w1,50):.2e}/{np.percentile(w1,99):.2e}/{w1.max():.2e} m/s")
    for dt in a.dt:
        c = w1 * dt / dz_top
        i = int(np.argmax(c)); la = lat[wet][i]; lo = lon[wet][i]
        print(f"  dt={dt:6.0f}s: CFL p50 {np.percentile(c,50):.3f} p99 {np.percentile(c,99):.3f} "
              f"max {c.max():.3f} at ({la:.1f}N, {lo:.1f}E); n(CFL>0.5)={int((c>0.5).sum())} n(CFL>1)={int((c>1).sum())}")


if __name__ == "__main__":
    main()
