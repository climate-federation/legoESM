"""Plot the MULTI-NODE banded-MG barotropic wall-time (user: use multi-node).

Parses the ``baro_mcut_multinode.sbatch`` log (job 8488551) — one WALL-TIME block
per rank count — and plots (a) jacobi-M60 vs multigrid-M12 ms/solve at each node
count, and (b) the MG wall-time speedup vs node count.  The reduction-latency
wall (Gloo allreduce) grows with node count, so the MG's 24-reductions/step
(M=12) beats jacobi's 120 (M=60) by a margin that GROWS np8(1 node)->np32(4) —
the reduction-latency-wall lever's payoff at the scale that matters.

Usage::

    python scripts/plot/plot_baro_mcut_multinode.py \
        --log results/scaling_ginsburg/logs/baro_mcut_mn_<JOBID>.out \
        --out docs/scaling/baro_mcut_multinode.png
"""
from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse(log_path):
    """Return {np: (jacobi_ms, mg_ms, speedup)} from the WALL-TIME blocks."""
    txt = open(log_path).read()
    out = {}
    # Each block: "=== WALL-TIME npN (...)" then jacobi/multigrid ms lines.
    for blk in re.split(r"=== WALL-TIME np", txt)[1:]:
        m_np = re.match(r"(\d+)", blk)
        m_j = re.search(r"jacobi\s+M=\d+:\s+([\d.]+)\s+ms/solve", blk)
        m_g = re.search(r"multigrid\s+M=\d+:\s+([\d.]+)\s+ms/solve", blk)
        if m_np and m_j and m_g:
            nr = int(m_np.group(1))
            tj, tg = float(m_j.group(1)), float(m_g.group(1))
            out[nr] = (tj, tg, tj / tg)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="docs/scaling/baro_mcut_multinode.png")
    args = p.parse_args()

    data = _parse(args.log)
    if not data:
        raise SystemExit(f"no WALL-TIME blocks parsed from {args.log}")
    nps = sorted(data)
    jac = [data[n][0] for n in nps]
    mg = [data[n][1] for n in nps]
    spd = [data[n][2] for n in nps]
    nodes = [max(1, n // 8) for n in nps]   # 8 ranks/node
    xlabels = [f"np{n}\n({nd} node{'s' if nd > 1 else ''})"
               for n, nd in zip(nps, nodes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    x = range(len(nps))
    w = 0.36
    ax1.bar([i - w / 2 for i in x], jac, w, label="jacobi M=60 (120 red/step)",
            color="#d62728")
    ax1.bar([i + w / 2 for i in x], mg, w, label="multigrid M=12 (24 red/step)",
            color="#1f77b4")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("ms / barotropic solve (lower better)")
    ax1.set_title("Banded-MG vs Jacobi barotropic solve — multi-node\n"
                  "LL192x384, Ginsburg CPU-MPI (8 ranks/node)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(list(x), spd, "o-", color="#1f77b4", lw=2)
    for i, s in zip(x, spd):
        ax2.annotate(f"{s:.2f}x", (i, s), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("MG wall-time speedup vs jacobi")
    ax2.set_title("Reduction-latency win GROWS with node count\n"
                  "(Gloo allreduce latency x rank dominates)", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for n in nps:
        tj, tg, s = data[n]
        print(f"  np{n}: jacobi {tj:.2f} ms, multigrid {tg:.2f} ms, {s:.2f}x")


if __name__ == "__main__":
    main()
