"""Weak-scaling view of the banded-MG barotropic M-cut (task #27).

Overlays the residual-vs-M curves from two (or more) ``bench_barotropic_mcut.py``
CSVs taken at FIXED rows-per-rank (e.g. np2/96x192 and np4/192x384, both 48 lat
rows/rank).  The multinode WEAK-SCALING claim is that the banded multigrid's
iteration count M to reach a target residual stays ~CONSTANT as ranks+problem
grow (O(log n) convergence + a reduction-free V-cycle), so its curves coincide
across rank counts — while Jacobi's per-step reduction count to make progress
does not improve.  The per-step global-reduction count is 2*M.

Usage::

    python scripts/plot/plot_barotropic_mcut_weakscale.py \
        --csv docs/scaling/barotropic_mcut_np2.csv \
              docs/scaling/barotropic_mcut_np4.csv \
        --out docs/scaling/barotropic_mcut_weakscale.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    series = defaultdict(list)        # name -> [(M, rel)]
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            series[row["preconditioner"]].append(
                (int(row["M"]), float(row["rel_residual"])))
            meta = row
    for k in series:
        series[k].sort()
    return series, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--out", default="docs/scaling/barotropic_mcut_weakscale.png")
    p.add_argument("--target", type=float, default=1e-6)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    base = {"jacobi": "Reds", "multigrid": "Blues",
            "zonal_line": "Greens", "chebyshev": "Purples"}
    # darker line per larger rank count
    loaded = [(_load(c)) for c in args.csv]
    nps = sorted({int(m.get("n_ranks", 0)) for _, m in loaded})
    shade = {n: 0.45 + 0.5 * (i / max(1, len(nps) - 1))
             for i, n in enumerate(nps)}
    summary = []
    for series, meta in loaded:
        nr = int(meta.get("n_ranks", 0))
        gl = f"{meta.get('n_lat')}x{meta.get('n_lon')}"
        for name, pts in series.items():
            Ms = [m for m, _ in pts]
            rs = [r for _, r in pts]
            cmap = plt.get_cmap(base.get(name, "Greys"))
            ax.semilogy(Ms, rs, "o-", color=cmap(shade[nr]), lw=2,
                        label=f"{name} np{nr} ({gl})")
            hit = [m for m, r in pts if r <= args.target]
            if hit:
                summary.append((name, nr, min(hit)))

    ax.axhline(args.target, color="gray", ls="--", lw=1,
               label=f"target {args.target:.0e}")
    ax.set_xlabel("outer PCG iterations  M  (per-step global reductions = 2·M)")
    ax.set_ylabel("relative residual")
    mg = sorted([(nr, m) for (nm, nr, m) in summary if nm == "multigrid"])
    sub = ("multigrid M→target: " +
           ", ".join(f"np{nr}={m}" for nr, m in mg)) if mg else ""
    rank_indep = (len({m for _, m in mg}) == 1) if mg else False
    ax.set_title(
        "Barotropic-PCG M-cut — WEAK SCALING (fixed rows/rank)\n"
        + sub + ("  → rank-independent ✓" if rank_indep else ""),
        fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for nm, nr, m in sorted(summary):
        print(f"  {nm:10s} np{nr}: M->target={m} (reductions/step={2 * m})")


if __name__ == "__main__":
    main()
