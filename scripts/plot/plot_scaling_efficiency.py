"""Parallel-efficiency scaling plot — the clean, standard presentation of
weak+strong multi-node scaling (vs the raw-throughput view whose different
magnitudes + halo-latency dips read as "messy").

Efficiency normalises every curve to its own smallest device count N0:

    E(N) = mcells_per_s(N) * N0 / (N * mcells_per_s(N0))

For STRONG scaling (fixed total, throughput ∝ 1/time) this is speedup/N; for
WEAK scaling (cells ∝ N, ideal throughput ∝ N) it is the same ratio.  Ideal =
1.0 (flat black line).  Reads the tidy CSV from ``aggregate_bcw_scaling.py``;
2x2 [component x mode] panels, COLOR = grid, LINESTYLE = precision (solid f64 /
dashed f32), MARKER = backend.

    python scripts/plot/plot_scaling_efficiency.py \\
        --csv results/scaling_ginsburg/multinode_clean/clean_tidy.csv --out docs/scaling
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_COLOR = {"icosahedral": "#1f77b4", "cubed-sphere": "#2ca02c",
              "latlon": "#d62728", "spectral": "#9467bd"}
PREC_LS = {"float64": "-", "float32": "--"}
BACK_MK = {"CPU": "o", "GPU": "s"}
PANELS = [("atm", "strong"), ("atm", "weak"),
          ("ocean", "strong"), ("ocean", "weak")]


def _read(csv_path: Path) -> list[dict]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def _group(rows):
    """(component, mode_class) -> (grid, precision, backend) -> {n_devices: mc}.
    Dedup by (n_devices): keep the max mc if duplicates."""
    g = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        try:
            nd = int(r["n_devices"])
            mc = float(r["mcells_per_s"])
        except (ValueError, KeyError, TypeError):
            continue
        if nd <= 0 or not math.isfinite(mc) or mc <= 0:
            continue
        mode = "strong" if r.get("mode") == "strong" else "weak"
        key = (r.get("grid"), r.get("precision"), r.get("backend"))
        d = g[(r.get("component"), mode)][key]
        d[nd] = max(d.get(nd, 0.0), mc)
    return g


def efficiency_curve(nd_to_mc: dict[int, float]) -> list[tuple[int, float]]:
    """[(n_devices, efficiency)] normalised to the smallest device count."""
    if not nd_to_mc:
        return []
    n0 = min(nd_to_mc)
    mc0 = nd_to_mc[n0]
    out = []
    for n in sorted(nd_to_mc):
        if mc0 > 0:
            out.append((n, nd_to_mc[n] * n0 / (n * mc0)))
    return out


def make_figure(rows, out_dir: Path) -> Path:
    g = _group(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (comp, mode) in zip(axes.flat, PANELS):
        curves = g.get((comp, mode), {})
        all_n = set()
        for (grid, prec, backend), nd_to_mc in sorted(curves.items()):
            pts = efficiency_curve(nd_to_mc)
            if len(pts) < 1:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            all_n.update(xs)
            ax.plot(xs, ys, marker=BACK_MK.get(backend, "x"),
                    color=GRID_COLOR.get(grid, "#777777"),
                    linestyle=PREC_LS.get(prec, ":"),
                    label=f"{grid} {prec[-2:]} {backend}")
        if all_n:
            lo, hi = min(all_n), max(all_n)
            ax.plot([lo, hi], [1.0, 1.0], "k--", alpha=0.5, lw=1, label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_ylim(0.0, 1.25)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(f"{comp.upper()} — {mode} scaling efficiency")
        ax.set_xlabel("devices (MPI ranks / nodes)")
        ax.set_ylabel("parallel efficiency (1 = ideal)")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Parallel efficiency by grid — weak+strong, f32 (dashed) + f64 "
                 "(solid); 1.0 = ideal scaling", y=1.0, fontsize=12)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "scaling_efficiency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="docs/scaling")
    args = p.parse_args()
    rows = _read(Path(args.csv))
    out = make_figure(rows, Path(args.out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
