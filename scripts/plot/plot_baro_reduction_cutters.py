"""Plot the multi-node barotropic reduction-cutter comparison (job 8488773).

Phase-split (job 8488677) found the barotropic reduction-latency wall is the #1
multi-node bottleneck at np32 (24% of step, 120-allreduce floor 21%).  This plots
the candidate cutters' barotropic-solve speedup vs jacobi-standard across node
count.  CHEBYSHEV (reduction-free degree-4 inner polynomial, M60->20 = 40 vs 120
reductions, cheap matvecs) WINS 1.23x at np16/np32 — where the banded MG
(expensive zonal-line V-cycle) LOST.  single_reduce (120->60, same per-iter) is
marginal; MG is wall-time-negative.

Parses the ``baro_reduction_cutters_mn.sbatch`` log.

Usage::

    python scripts/plot/plot_baro_reduction_cutters.py \
        --log results/scaling_ginsburg/logs/baro_redcut_mn_<JOBID>.out \
        --out docs/scaling/baro_reduction_cutters.png
"""
from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = ["jacobi_std", "jacobi_singlereduce", "chebyshev", "multigrid"]
COLORS = {"jacobi_std": "#7f7f7f", "jacobi_singlereduce": "#2ca02c",
          "chebyshev": "#1f77b4", "multigrid": "#d62728"}


def _parse(log_path):
    """{np: {config: speedup}} from the WALL-TIME blocks."""
    txt = open(log_path).read()
    out = {}
    for blk in re.split(r"=== WALL-TIME np", txt)[1:]:
        m_np = re.match(r"(\d+)", blk)
        if not m_np:
            continue
        nr = int(m_np.group(1))
        d = {}
        for cfg in CONFIGS:
            m = re.search(rf"{cfg}\s+M=\s*\d+:\s+[\d.]+ ms/solve.*?speedup=([\d.]+)x",
                          blk)
            if m:
                d[cfg] = float(m.group(1))
        if d:
            out[nr] = d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="docs/scaling/baro_reduction_cutters.png")
    args = p.parse_args()

    data = _parse(args.log)
    if not data:
        raise SystemExit(f"no WALL-TIME blocks parsed from {args.log}")
    nps = sorted(data)
    nodes = [max(1, n // 8) for n in nps]
    xlabels = [f"np{n}\n({nd} node{'s' if nd > 1 else ''})"
               for n, nd in zip(nps, nodes)]
    x = list(range(len(nps)))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for cfg in CONFIGS:
        ys = [data[n].get(cfg, float("nan")) for n in nps]
        ax.plot(x, ys, "o-", color=COLORS[cfg], lw=2,
                label=cfg.replace("_", " "))
    ax.axhline(1.0, color="black", ls="--", lw=1, label="jacobi baseline")
    # annotate chebyshev (the winner)
    for i, n in enumerate(nps):
        s = data[n].get("chebyshev")
        if s:
            ax.annotate(f"{s:.2f}x", (i, s), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9,
                        color=COLORS["chebyshev"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("barotropic-solve speedup vs jacobi-standard (>1 = win)")
    ax.set_title(
        "Multi-node barotropic reduction-cutters (LL192x384, job 8488773)\n"
        "CHEBYSHEV wins 1.23x at np16/np32 (reduction-free inner, cheap matvecs)\n"
        "where the banded MG LOST (V-cycle compute). single_reduce marginal.",
        fontsize=9)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for n in nps:
        print(f"  np{n}: " + ", ".join(
            f"{c}={data[n].get(c, float('nan')):.2f}x" for c in CONFIGS))


if __name__ == "__main__":
    main()
