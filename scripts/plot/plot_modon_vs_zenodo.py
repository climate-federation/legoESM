#!/usr/bin/env python
"""Side-by-side: our duo-stepper colliding-modon run vs the Zenodo C48
case-8 reference (their atmos_daily.nc daily means, already on the 1deg
grid our runner samples).

Compares wind-speed pattern + amplitude at matched days.  PASS = two
coherent dipoles that approach, collide near lon 180 (~day 15-20), and
depart meridionally toward the poles (day 30), with max wind held in
the reference's ~15-20 m/s band — i.e. NOT dispersed.

Usage: plot_modon_vs_zenodo.py --run <our.npz> --out <panels.png>
                               [--days 0,10,20,30]
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="our runner npz")
    ap.add_argument("--ref", default="/burg-archive/glab/users/pg2328/"
                    "Code/FV3/duogrid_zenodo/extracted/Code and "
                    "simulations files/C48.sw.case8.alpha0.duo.hord8/"
                    "rundir/atmos_daily.nc")
    ap.add_argument("--days", default="0,10,20,30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import netCDF4 as nc

    days = [int(x) for x in args.days.split(",")]
    z = np.load(args.run)
    t = z["times_days"]
    W = np.hypot(z["u"], z["v"])
    d = nc.Dataset(args.ref)
    ru = d.variables["ucomp"][:, 0]
    rv = d.variables["vcomp"][:, 0]
    rlat = d.variables["lat"][:]
    rlon = d.variables["lon"][:]
    RW = np.hypot(ru, rv)

    n = len(days)
    fig, ax = plt.subplots(n, 2, figsize=(15, 3.1 * n))
    for k, dd in enumerate(days):
        i = int(np.argmin(np.abs(t - dd)))
        our = W[i]
        ax[k][0].pcolormesh(z["lon"], z["lat"], our, cmap="viridis",
                            vmin=0, vmax=25)
        ax[k][0].set_ylim(-55, 55)
        ax[k][0].set_title(f"ours (duo case-8) day {t[i]:.0f} — "
                           f"max {our.max():.1f} m/s (instant)",
                           fontsize=10)
        rref = RW[min(dd, RW.shape[0] - 1)]
        im = ax[k][1].pcolormesh(rlon, rlat, rref, cmap="viridis",
                                 vmin=0, vmax=25)
        ax[k][1].set_ylim(-55, 55)
        ax[k][1].set_title(f"Zenodo C48 day {dd} — max {rref.max():.1f} "
                           "m/s (daily mean)", fontsize=10)
    plt.colorbar(im, ax=ax, label="wind speed (m/s)", shrink=0.6)
    fig.suptitle("Colliding modons: bounded duo stepper vs Zenodo "
                 f"reference — {args.run.split('/')[-1]}", fontsize=12)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print("saved", args.out)
    # numeric verdict
    print("day   ours_max   ref_max")
    for dd in days:
        i = int(np.argmin(np.abs(t - dd)))
        print(f"{dd:4d}  {W[i].max():8.2f}   "
              f"{RW[min(dd, RW.shape[0]-1)].max():7.2f}")


if __name__ == "__main__":
    main()
