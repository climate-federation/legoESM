"""Plot the barotropic-PCG iteration-count (M) cut: jacobi vs banded multigrid.

Reads the CSV from ``scripts/bench/bench_barotropic_mcut.py`` and draws
rel_residual vs outer-iteration M (log-y) for each preconditioner, with the
target residual and the per-step GLOBAL-reduction count (2*M) annotated.  The
gap between the curves at a fixed residual IS the reduction-latency win on the
multinode barotropic solve.

Usage::

    python scripts/plot/plot_barotropic_mcut.py \
        --csv docs/scaling/barotropic_mcut.csv --out docs/scaling/barotropic_mcut.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="docs/scaling/barotropic_mcut.csv")
    p.add_argument("--out", default="docs/scaling/barotropic_mcut.png")
    p.add_argument("--target", type=float, default=1e-6)
    args = p.parse_args()

    series = defaultdict(list)        # name -> [(M, rel)]
    meta = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            series[row["preconditioner"]].append(
                (int(row["M"]), float(row["rel_residual"])))
            meta = row
    for k in series:
        series[k].sort()

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = {"jacobi": "#d62728", "multigrid": "#1f77b4",
              "zonal_line": "#2ca02c", "chebyshev": "#9467bd"}
    m_to_target = {}
    for name, pts in series.items():
        Ms = [m for m, _ in pts]
        rs = [r for _, r in pts]
        ax.semilogy(Ms, rs, "o-", color=colors.get(name, "k"), label=name, lw=2)
        hit = [m for m, r in pts if r <= args.target]
        if hit:
            m_to_target[name] = min(hit)

    ax.axhline(args.target, color="gray", ls="--", lw=1,
               label=f"target {args.target:.0e}")
    for name, m in m_to_target.items():
        ax.axvline(m, color=colors.get(name, "k"), ls=":", lw=1, alpha=0.6)

    title = "Barotropic-PCG M-cut: banded multigrid vs Jacobi"
    if meta:
        title += (f"\n{meta.get('n_lat')}x{meta.get('n_lon')} lat-lon, "
                  f"np{meta.get('n_ranks')} (MPI band); reductions/step = 2·M")
    if "jacobi" in m_to_target and "multigrid" in m_to_target:
        mj, mg = m_to_target["jacobi"], m_to_target["multigrid"]
        title += (f"\nM-cut {mj}->{mg} = {mj / mg:.1f}x  |  "
                  f"reductions/step {2 * mj}->{2 * mg}")
    ax.set_xlabel("outer PCG iterations  M  (per-step global reductions = 2·M)")
    ax.set_ylabel("relative residual")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
