"""Where the atmosphere step's time actually goes, per lane, per device count.

Each bar is a measured step split into local work and communication, with
the perfect-scaling time marked. The split comes from paired arms in which
every halo collective is deleted and nothing else changes, because the
profiler on this stack does not record these collectives at all:

    communication = full - (same program with the collectives removed)

Numbers are literals here rather than parsed, because they come from A/B
job logs rather than a single receipt stream; the SLURM job id of every
arm is carried next to it so any bar can be traced.

Usage
-----
    python scripts/plot/plot_step_budget_2lanes.py --out step_budget.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (devices, full ms, no-comm ms, ideal ms, job id)
# ideal = single-device step / devices, same build and flags.
LATLON = [
    (32, 6.980, 6.304, 201.4807 / 32, "27069398"),
    (64, 4.851, 3.368, 201.4807 / 64, "27069398"),
]
MPAS = [
    (8, 23.135, 23.055, 173.84 / 8, "27068830"),
    (16, 19.230, 19.120, 173.84 / 16, "27068832"),
    (32, 7.230, 7.190, 173.84 / 32, "27068833"),
]

LANES = [
    ("lat-lon 2048x4096 x 26 lev", LATLON),
    ("icosahedral 2.6M cells x 26 lev", MPAS),
]

C_LOCAL = "#0072B2"
C_COMM = "#D55E00"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step_budget.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, (title, rows) in zip(axes, LANES):
        x = np.arange(len(rows))
        local = np.array([r[2] for r in rows])
        # A measured communication term can come out very slightly negative
        # when it is zero to within the arm spread; clamp for drawing only
        # and say so in the annotation.
        comm = np.array([max(0.0, r[1] - r[2]) for r in rows])
        ideal = np.array([r[3] for r in rows])

        ax.bar(x, local, color=C_LOCAL, label="local work")
        ax.bar(x, comm, bottom=local, color=C_COMM, label="communication")
        ax.plot(x, ideal, "k_", ms=26, mew=2.0, label="perfect scaling")

        for i, r in enumerate(rows):
            pct = 100.0 * comm[i] / r[1]
            ax.annotate(f"{pct:.0f}% comm", (i, r[1]), ha="center",
                        va="bottom", fontsize=8.5,
                        xytext=(0, 3), textcoords="offset points")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r[0]}" for r in rows])
        ax.set_xlabel("GPUs")
        ax.set_ylabel("ms / step")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Measured step budget: local work vs communication",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    for title, rows in LANES:
        print(title)
        for dev, full, nc, ideal, job in rows:
            comm = full - nc
            print(f"  {dev:>4} GPU  full {full:7.3f}  local {nc:7.3f}  "
                  f"comm {comm:+7.3f} ({100*comm/full:5.1f}%)  "
                  f"ideal {ideal:6.3f}  job {job}")


if __name__ == "__main__":
    main()
