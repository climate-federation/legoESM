"""Horizontal cross-section grid for a GABLS1 spectral-LES run.

Reads the ``snap_NNN.npz`` frames written by ``scripts/run/les_record.py`` and
lays out horizontal (x-y) snapshots as a HEIGHT x TIME grid for one field
(default vertical velocity ``w`` — the clearest signature of the resolved SBL
turbulence). One PNG per field.

Usage
-----
.. code-block:: bash

   .venv/bin/python scripts/plot/plot_gabls1_slices.py \\
       --run results/les_gabls1_gpu --field w --times 4 --out results/les_gabls1_gpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CMAP = {"w": "RdBu_r", "u": "viridis", "v": "RdBu_r", "theta": "inferno"}
_LABEL = {"w": "w [m/s]", "u": "u [m/s]", "v": "v [m/s]", "theta": r"$\theta$ [K]"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="run dir (holds snapshots/)")
    p.add_argument("--field", choices=list(_CMAP), default="w")
    p.add_argument("--times", type=int, default=4, help="number of time columns")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.out or args.run

    snaps = sorted((args.run / "snapshots").glob("snap_*.npz"))
    if not snaps:
        raise SystemExit(f"no snapshots in {args.run}/snapshots")
    # Pick `times` frames evenly across the run (skip the very first = laminar IC).
    pick = np.unique(np.linspace(1, len(snaps) - 1, args.times).round().astype(int))
    frames = [snaps[i] for i in pick]

    d0 = np.load(frames[0])
    heights = d0["heights"]
    n_h = len(heights)
    Lx, Ly = float(d0["Lx"]), float(d0["Ly"])
    extent = [0, Lx, 0, Ly]

    fld = args.field
    # Shared symmetric (w, v) or common (theta, u) color scale across the grid.
    stack = np.array([np.load(f)[fld] for f in frames])  # (T, n_h, ny, nx)
    if fld in ("w", "v"):
        a = np.percentile(np.abs(stack), 99)
        vmin, vmax = -a, a
    else:
        vmin = np.percentile(stack, 1)
        vmax = np.percentile(stack, 99)

    n_t = len(frames)
    fig, axes = plt.subplots(n_h, n_t, figsize=(2.6 * n_t + 1.2, 2.6 * n_h),
                             squeeze=False)
    # Rows = height (descending: high at top), cols = time.
    im = None
    for r in range(n_h):
        hi = n_h - 1 - r  # height index (top row = highest)
        for c, f in enumerate(frames):
            d = np.load(f)
            ax = axes[r][c]
            im = ax.imshow(d[fld][hi], origin="lower", extent=extent,
                           cmap=_CMAP[fld], vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {float(d['t_hours']):.2f} h", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"z = {heights[hi]:.0f} m", fontsize=10)
    fig.suptitle(f"GABLS1 LES (dynamic LASD SGS) — horizontal {_LABEL[fld]} "
                 f"slices  [{stack.shape[-1]}x{stack.shape[-2]}]", fontsize=12)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label(_LABEL[fld])
    out = out_dir / f"gabls1_slices_{fld}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
