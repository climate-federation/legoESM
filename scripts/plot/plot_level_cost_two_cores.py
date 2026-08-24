"""Cost per level against level count, for both atmosphere cores.

One GPU, no communication, float32, 60 timed steps per arm, two replicates.
The unstructured core varies threefold with the level count and in no pattern
a rule can express; the lat-lon core over the same counts is flat. That
contrast is the finding: the effect belongs to the unstructured kernels, not
to the compiler or the hardware.

Numbers are literals with their job ids beside them, because they come from
several A/B job logs rather than one receipt stream.

Usage
-----
    python scripts/plot/plot_level_cost_two_cores.py --out level_cost.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# level count -> picoseconds per cell (or column) per level.
# Icosahedral: 163,842 cells, jobs 27070658 / 27073083 / 27073084 / 27074125 /
# 27074126 / 27074455 / 27076677.
ICOS = {13: 3520, 16: 1547, 18: 2318, 20: 1524, 21: 2578, 22: 4910, 24: 4147,
        25: 4224, 26: 4755, 27: 3861, 28: 4058, 30: 4591, 31: 3741, 32: 1629,
        34: 2949, 36: 1706, 40: 1664, 52: 1621}
# lat-lon: 1024x2048 columns, job 27078027.
LATLON = {16: 851, 20: 900, 22: 907, 26: 882, 30: 890, 32: 891, 36: 954,
          40: 937}

PRODUCTION_LEVELS = 26
CHEAP_BAR = 1800   # icosahedral counts at or under this are the cheap set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="level_cost.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)

    for ax, (title, data, unit) in zip(axes, [
            ("icosahedral core (163,842 cells)", ICOS, "cell"),
            ("lat–lon core (1024x2048 columns)", LATLON, "column")]):
        ks = sorted(data)
        vs = [data[k] for k in ks]
        colors = ["#0072B2" if data[k] <= CHEAP_BAR else "#D55E00" for k in ks]
        ax.bar([str(k) for k in ks], vs, color=colors)
        ax.set_xlabel("vertical levels")
        ax.set_ylabel(f"picoseconds per {unit} per level")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        lo, hi = min(vs), max(vs)
        ax.annotate(f"spread {hi / lo:.1f}x", xy=(0.02, 0.93),
                    xycoords="axes fraction", fontsize=9)
        if PRODUCTION_LEVELS in data:
            i = ks.index(PRODUCTION_LEVELS)
            ax.annotate("production", xy=(i, data[PRODUCTION_LEVELS]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8)

    fig.suptitle("Cost per level, same code, only the level count changed",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    for name, data in (("icosahedral", ICOS), ("lat-lon", LATLON)):
        lo, hi = min(data.values()), max(data.values())
        print(f"{name}: {lo}-{hi} ps, spread {hi / lo:.2f}x, "
              f"production {data.get(PRODUCTION_LEVELS)} "
              f"({data.get(PRODUCTION_LEVELS, lo) / lo:.2f}x cheapest)")


if __name__ == "__main__":
    main()
