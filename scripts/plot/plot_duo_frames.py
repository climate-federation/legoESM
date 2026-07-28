#!/usr/bin/env python
"""Wind-speed snapshot panels from a duo-stepper frames npz.

Input: the run_duo_stepper_modon/--out npz (times_days, u, v on the
1-degree c2l lens grid).  Optionally a second npz (--ref, e.g. the
extracted Zenodo case-8 reference frames) rendered as a bottom row at
the nearest available times.  Writes one PNG.

Usage:
  plot_duo_frames.py --frames m.npz --days 1,5,10 --out snap.png \
      [--ref zenodo_case8_c48_ref.npz] [--vmax 30]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pick(times, day):
    k = int(np.argmin(np.abs(np.asarray(times) - day)))
    return k, float(times[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--days", required=True,
                    help="comma-separated days to render")
    ap.add_argument("--ref", default=None,
                    help="optional reference frames npz (bottom row)")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    days = [float(x) for x in args.days.split(",")]
    d = np.load(args.frames, allow_pickle=False)
    w = np.hypot(d["u"], d["v"])
    rows = [("ours", d["times_days"], w)]
    if args.ref:
        r = np.load(args.ref, allow_pickle=False)
        tk = next((k for k in ("times_days", "times", "days")
                   if k in r.files), None)
        if "u" in r.files and "v" in r.files:
            wr = np.hypot(r["u"], r["v"])
        else:
            wk = next((k for k in ("wind", "speed", "w")
                       if k in r.files), None)
            if wk is None or tk is None:
                raise SystemExit(
                    f"ref {args.ref}: no usable keys; has {r.files}")
            wr = r[wk]
        rows.append(("oracle", r[tk], wr))

    vmax = args.vmax
    if vmax is None:
        vmax = max(float(np.nanmax(w[pick(rows[0][1], dd)[0]]))
                   for dd in days)

    nr, nc = len(rows), len(days)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 2.6 * nr),
                             squeeze=False, constrained_layout=True)
    for i, (label, times, ww) in enumerate(rows):
        for j, dd in enumerate(days):
            k, tk_ = pick(times, dd)
            ax = axes[i][j]
            im = ax.imshow(ww[k], origin="lower",
                           extent=(0, 360, -90, 90),
                           vmin=0, vmax=vmax, cmap="viridis",
                           aspect="auto")
            ax.set_title(f"{label} d{tk_:g}  max {np.nanmax(ww[k]):.1f}",
                         fontsize=9)
            if j == nc - 1:
                fig.colorbar(im, ax=ax, shrink=0.85, label="|V| m/s")
    fig.suptitle(f"{args.frames.split('/')[-1]}", fontsize=10)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
