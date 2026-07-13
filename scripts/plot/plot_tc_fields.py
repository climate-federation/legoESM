"""Horizontal field montage for the rotating-RCE tropical-cyclone experiment.

Reads the ``sfc_*.npz`` snapshots written by ``rce_snapshot.save_surface_levels``
and lays out FIELD (rows) x TIME (cols): surface wind speed (+vectors),
mid-level temperature anomaly (warm-core diagnostic), column water vapour,
surface precipitation. One PNG.

Usage::

    .venv/bin/python scripts/plot/plot_tc_fields.py --run results/tc_rotating_rce \\
        --t0-day 60 --times 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--t0-day", type=float, default=60.0,
                   help="sim-day at restart, subtracted so the axis is t+days.")
    p.add_argument("--times", type=int, default=4)
    p.add_argument("--level-km", type=float, default=5.0,
                   help="height [km] for the temperature-anomaly row.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    # --out may name a *.png file OR a directory (montage -> <dir>/tc_fields.png);
    # create the parent either way so it never silently fails or nests a dir.
    if args.out is not None and args.out.suffix:
        out = args.out
    else:
        out = (args.out or args.run) / "tc_fields.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    snaps = sorted((args.run / "snapshots").glob("sfc_*.npz"))
    if not snaps:
        raise SystemExit(f"no snapshots in {args.run}/snapshots")
    pick = np.unique(np.linspace(0, len(snaps) - 1, args.times).round().astype(int))
    frames = [snaps[i] for i in pick]

    d0 = np.load(frames[0])
    heights = d0["heights"]
    klev = int(np.argmin(np.abs(heights - args.level_km * 1e3)))
    Lx = float(d0["Lx"]) / 1e3
    ext = [0, Lx, 0, Lx]

    # shared scales across the row
    def stack(key, fn):
        return np.array([fn(np.load(f)) for f in frames])
    wind = stack(None, lambda d: np.hypot(d["u_sfc"], d["v_sfc"]))
    Tanom = stack(None, lambda d: d["T_levels"][klev] - d["T_levels"][klev].mean())
    cwv = stack(None, lambda d: d["cwv"])
    precip = stack(None, lambda d: d["precip"])

    rows = [
        ("surface wind |U| [m/s]", wind, "viridis", (0, max(20, np.percentile(wind, 99)))),
        (f"T' at {heights[klev]/1e3:.0f} km [K]", Tanom, "RdBu_r",
         (-np.percentile(np.abs(Tanom), 99), np.percentile(np.abs(Tanom), 99))),
        ("column water vapour [mm]", cwv, "BrBG", (np.percentile(cwv, 2), np.percentile(cwv, 98))),
        ("surface precip [mm/day]", precip, "Blues", (0, np.percentile(precip, 99.5))),
    ]
    nt = len(frames)
    fig, axes = plt.subplots(len(rows), nt, figsize=(3.0 * nt + 1.5, 3.0 * len(rows)),
                             squeeze=False)
    for r, (label, data, cmap, (vmin, vmax)) in enumerate(rows):
        for c, f in enumerate(frames):
            d = np.load(f)
            ax = axes[r][c]
            im = ax.imshow(data[c], origin="lower", extent=ext, cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t + {float(d['day']) - args.t0_day:.2f} d", fontsize=11)
                # wind vectors
                u, v = d["u_sfc"], d["v_sfc"]
                s = max(1, u.shape[0] // 16)
                yy, xx = np.mgrid[0:u.shape[0], 0:u.shape[1]]
                ax.quiver((xx[::s, ::s] + 0.5) * d["dx"] / 1e3,
                          (yy[::s, ::s] + 0.5) * d["dx"] / 1e3,
                          u[::s, ::s], v[::s, ::s], color="white",
                          scale=400, width=0.004)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
        fig.colorbar(im, ax=axes[r], shrink=0.7, pad=0.01)
    fig.suptitle("Rotating RCE (f-plane, 512 km, gray-RCE restart) — TC fields", fontsize=13)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
