"""Barotropic-solver head-to-head: implicit_cn vs explicit_substep+local-clamp.

The campaign's biggest multi-node ocean result (2026-06-15, SOTA directive): the
MOM6/MPAS-Ocean-style split-explicit barotropic with a reduction-free local
eta-floor clamp BEATS the implicit Crank-Nicolson PCG at >=2 nodes, and the win
GROWS with node count because implicit_cn anti-scales on the 120-allreduce wall.
Same-job measurements (LL192 f64, Ginsburg CPU-MPI, 8 ranks/node).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (np, ms/step) — same-job head-to-heads (jobs 8498971 np8/16/32, 8499266 np64).
IMPLICIT = [(8, 133.94), (16, 77.44), (32, 64.73), (64, 84.21)]
EXPLICIT = [(8, 143.91), (16, 67.44), (32, 52.52), (64, 50.92)]


def _eff(series):
    n0, t0 = series[0]
    return [(t0 / t) / (n / n0) for n, t in series]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/scaling_ginsburg/baro_solver_h2h.png")
    args = p.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    nodes_i = [n // 8 for n, _ in IMPLICIT]
    nodes_e = [n // 8 for n, _ in EXPLICIT]

    # Panel 1: ms/step
    ax1.plot(nodes_i, [t for _, t in IMPLICIT], "o-", color="#d62728", lw=2,
             label="implicit_cn (M=60 PCG, 120 allreduce)")
    ax1.plot(nodes_e, [t for _, t in EXPLICIT], "s-", color="#1f77b4", lw=2,
             label="explicit_substep + local-clamp (SOTA, reduction-free)")
    for n, t in IMPLICIT:
        ax1.annotate(f"{t:.0f}", (n // 8, t), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="#d62728")
    for n, t in EXPLICIT:
        ax1.annotate(f"{t:.0f}", (n // 8, t), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=8, color="#1f77b4")
    ax1.set_xticks([1, 2, 4, 8])
    ax1.set_xticklabels(["1\n(np8)", "2\n(np16)", "4\n(np32)", "8\n(np64)"])
    ax1.set_xlabel("nodes (8 ranks/node)")
    ax1.set_ylabel("ms / step  (lower better)")
    ax1.set_title("Ocean full-step ms/step (LL192 f64, same-job)\n"
                  "implicit anti-scales np32->np64; explicit holds", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: strong efficiency
    ax2.plot(nodes_i, _eff(IMPLICIT), "o-", color="#d62728", lw=2, label="implicit_cn")
    ax2.plot(nodes_e, _eff(EXPLICIT), "s-", color="#1f77b4", lw=2,
             label="explicit+local-clamp")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="ideal")
    ax2.set_xticks([1, 2, 4, 8])
    ax2.set_xticklabels(["1", "2", "4", "8"])
    ax2.set_xlabel("nodes (8 ranks/node)")
    ax2.set_ylabel("strong-scaling efficiency")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Strong efficiency: explicit 0.35 vs implicit 0.20 at np64\n"
                  "(crossover ~np16; win grows 1.15/1.23/1.65x)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SOTA split-explicit barotropic (MOM6/MPAS-O) beats implicit_cn "
                 "at scale — Ginsburg CPU-MPI", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
