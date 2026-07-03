"""Capstone multi-node scaling summary (2026-06-15, user 'use multi-node').

Three measured Ginsburg CPU-MPI curves (8 ranks/node), efficiency vs node count:
  * ocean STRONG (LL192 jacobi implicit_cn, job 8489477) — 0.88 np16, 0.55 np32
  * ocean WEAK (fixed ~590k cells/rank, job 8489559) — 0.97 np8->np16 (near-flat)
  * MPAS/icosahedral STRONG (I6 f64, job 8489516) — 0.84 np16, 0.69 np32

Honest takeaway: STRONG scaling degrades at np32 (fixed problem -> per-rank tile
shrinks -> the barotropic reduction-latency + baroclinic-halo Amdahl terms bite),
but WEAK scaling at production tile sizes is EXCELLENT (0.97) because compute
dominates the fixed reduction latency. Realistic big-per-rank production runs
scale well; the 'limit' is the small-tile strong regime on the Gloo/TCP fabric.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, color, [(np, ms)]) — measured, 8 ranks/node
SERIES = [
    ("ocean strong (LL192)", "#1f77b4",
     [(8, 139.85), (16, 79.12), (32, 63.06)]),
    ("MPAS strong (I6)", "#2ca02c",
     [(8, 99.31), (16, 59.27), (32, 36.00)]),
    ("ocean weak (~590k cells/rank)", "#d62728",
     [(8, 1422.95), (16, 1470.01)]),   # np32 OOM (LL6144 75M-cell global)
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/performance/scaling/multinode_summary.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for label, color, pts in SERIES:
        nps = [n for n, _ in pts]
        ms = [m for _, m in pts]
        base_np, base_ms = nps[0], ms[0]
        weak = "weak" in label
        # strong eff = (t0/t)/(np/np0); weak eff = t0/t
        eff = [(base_ms / m) / (n / base_np) if not weak else base_ms / m
               for n, m in zip(nps, ms)]
        nodes = [n // 8 for n in nps]
        ax.plot(nodes, eff, "o-", color=color, lw=2, label=label)
        for nd, e in zip(nodes, eff):
            ax.annotate(f"{e:.2f}", (nd, e), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=color)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="ideal")
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["1 node\n(np8)", "2 nodes\n(np16)", "4 nodes\n(np32)"])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("nodes (8 ranks/node)")
    ax.set_ylabel("scaling efficiency")
    ax.set_title(
        "Ginsburg multi-node scaling (CPU-MPI, 2026-06-15)\n"
        "STRONG degrades at 4 nodes (reduction/halo Amdahl, small tile); "
        "WEAK at production tile = 0.97 (compute-dominated)", fontsize=9)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
