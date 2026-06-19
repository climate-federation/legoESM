"""Ocean full-step multi-node STRONG scaling (job 8489477, LL192 jacobi).

The directive's weak/strong-scaling plot: full ocean step (implicit_cn jacobi,
production default) speedup + efficiency vs node count on Ginsburg CPU-MPI
(8 ranks/node).  Decent to np16 (2 nodes, 0.88 eff); degrades at np32 (4 nodes,
0.55 eff overall) — the reduction-latency wall (barotropic 120 allreduces/step =
24% of the np32 step) + the baroclinic halos that grow with rank, both
characterized as having no cheap wall-time fix
(docs/scaling/barotropic_multinode_verdict_2026-06-15.md).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# job 8489477, LL192x384, full ocean step (implicit_cn jacobi), ms/step
NP = [8, 16, 32]
NODES = [1, 2, 4]
MS = [139.85, 79.12, 63.06]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/scaling/ocean_strong_multinode.png")
    args = p.parse_args()

    base = MS[0]
    speedup = [base / m for m in MS]
    ideal = [n / NP[0] for n in NP]
    eff = [s / i for s, i in zip(speedup, ideal)]
    x = list(range(len(NP)))
    xl = [f"np{n}\n({nd} node{'s' if nd > 1 else ''})" for n, nd in zip(NP, NODES)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.plot(x, speedup, "o-", color="#1f77b4", lw=2, label="measured")
    ax1.plot(x, ideal, "k--", lw=1, label="ideal (linear)")
    for i, (s, m) in enumerate(zip(speedup, MS)):
        ax1.annotate(f"{s:.2f}x\n{m:.0f}ms", (i, s), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(xl)
    ax1.set_ylabel("speedup vs np8 (1 node)")
    ax1.set_title("Ocean full-step STRONG scaling — LL192, jacobi implicit_cn\n"
                  "Ginsburg CPU-MPI (8 ranks/node), job 8489477", fontsize=9)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(x, eff, "o-", color="#d62728", lw=2)
    for i, e in enumerate(eff):
        ax2.annotate(f"{e:.2f}", (i, e), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(x); ax2.set_xticklabels(xl)
    ax2.set_ylabel("strong-scaling efficiency")
    ax2.set_title("0.88 (np16) -> 0.55 (np32): reduction-latency + halo walls\n"
                  "(barotropic 120-allreduce + baroclinic halos; no cheap fix)",
                  fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for n, m, s, e in zip(NP, MS, speedup, eff):
        print(f"  np{n}: {m:.2f} ms, {s:.2f}x, eff {e:.2f}")


if __name__ == "__main__":
    main()
