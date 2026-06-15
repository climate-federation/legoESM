"""Plot the voronoi batched-halo A/B (legacy per-entity vs batched union-neighbor).

Grouped bars of ms/step at each rank count, legacy vs batched, with the % delta
annotated.  Overturns the stale "18x regression" verdict: at production scale
(icosahedral I6, f64) the batched halo (fewer, larger messages) is FASTER at
both single-node (np8) and multinode (np16).

Hard-coded from job 8488023 (icosahedral I6 nlev26 held_suarez f64); rerun the
A/B sbatch and update the dict to refresh.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# job 8488023, icosahedral I6 (1.06M cells), nlev26, held_suarez, f64, ms/step
DATA = {
    "np8 (1 node)":  {"legacy": 71.18, "batched": 67.90},
    "np16 (2 node)": {"legacy": 63.43, "batched": 61.23},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/scaling/voronoi_batched_ab.png")
    args = p.parse_args()

    labels = list(DATA)
    legacy = [DATA[k]["legacy"] for k in labels]
    batched = [DATA[k]["batched"] for k in labels]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    b1 = ax.bar(x - w / 2, legacy, w, label="legacy (per-entity)", color="#d62728")
    b2 = ax.bar(x + w / 2, batched, w, label="batched (union-neighbor)",
                color="#1f77b4")
    for i, k in enumerate(labels):
        d = 100.0 * (DATA[k]["batched"] - DATA[k]["legacy"]) / DATA[k]["legacy"]
        ax.annotate(f"{d:+.1f}%", (x[i] + w / 2, batched[i]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=10, color="#1f77b4", fontweight="bold")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms/step (lower better)")
    ax.set_title("Voronoi batched halo vs legacy — icosahedral I6, f64 (job 8488023)\n"
                 "batched (fewer/larger msgs) WINS at production scale — "
                 "overturns the stale I5/f32 '18x regression'", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
