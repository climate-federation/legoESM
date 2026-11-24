#!/usr/bin/env python
"""Per-site observed phi against |z/L|, with the model's stability functions."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOURS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
MARKERS = "osD^v<>Px*"
STYLES = {"dyer1974": ":", "beljaars_holtslag1991": "--", "grachev2007_sheba": "-.", "gryanik2020": (0, (5, 1))}


def curve(bins, sign):
    rows = [(abs(b[3]), b[5]) for b in bins if np.isfinite(b[3]) and b[3] * sign > 0]
    return np.array(sorted(rows)).T if rows else (None, None)


def main(path, out):
    r = json.load(open(path))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, kind, label in ((0, "m", r"$\phi_m$"), (1, "h", r"$\phi_h$")):
        for col, sign, name in ((0, -1, "unstable"), (1, 1, "stable")):
            ax = axes[row, col]
            for k, (site, s) in enumerate(r["sites"].items()):
                arm = s["offset_corrected"] if kind == "h" else s["primary"]
                entry = arm["site"].get(kind)
                if entry is None:
                    continue
                tag = site
                x, y = curve(entry["bins"], sign)
                if x is not None:
                    ax.plot(x, y, marker=MARKERS[k % len(MARKERS)], color=COLOURS[k % len(COLOURS)], lw=1, ms=5, label=tag)
            zeta = np.array(r["models"]["zeta"])
            sel = zeta * sign > 0
            for scheme, style in STYLES.items():
                ax.plot(np.abs(zeta[sel]), np.array(r["models"][scheme][kind])[sel], color="0.45", ls=style, lw=1.5, label=scheme)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{label}, {name}")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("|z/L|")
    seen = {}
    for ax in axes.flat:
        for handle, name in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(name, handle)
    h, l = list(seen.values()), list(seen)
    fig.legend(h, l, loc="center right", fontsize=8, frameon=False)
    fig.suptitle("Observed median per site, 30 min (SHEBA hourly), above canopy, gradients at most 50 m (no temperature-difference filter; heat only beyond |z/L| 0.1)\n$\\phi_h$ offset-corrected; d assumed 2/3 of canopy height at smear, chats", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.82, 0.95))
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:3])
