"""Throughput (Mcells/s) by grid type — atmosphere AND ocean, float32 AND
float64, strong AND weak scaling.

Reads the tidy CSV from ``aggregate_bcw_scaling.py`` and renders a 2x2
[component x mode] figure of throughput vs devices: COLOR = grid, LINESTYLE =
precision (solid float64 / dashed float32), MARKER = backend (o CPU / s GPU).
Also returns/prints a peak-throughput table per (component, mode, grid,
precision, backend).  Companion to ``plot_bcw_scaling.py`` (strong SYPD) and
``plot_multinode_summary.py`` (weak+strong efficiency).

    python scripts/plot/plot_throughput_by_grid.py \\
        --csv results/bcw_scaling/bcw_scaling_tidy.csv --out docs/scaling
"""
from __future__ import annotations

import argparse
import csv
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
    """(component, mode_class) -> (grid, precision, backend) -> [(n_devices, mc)].

    mode_class collapses weak/weak_band/weak_aspect into 'weak'."""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            nd = int(r["n_devices"])
            mc = float(r["mcells_per_s"])
        except (ValueError, KeyError, TypeError):
            continue
        if mc <= 0:
            continue
        mode = "strong" if r.get("mode") == "strong" else "weak"
        g[(r.get("component"), mode)][
            (r.get("grid"), r.get("precision"), r.get("backend"))
        ].append((nd, mc))
    return g


def peak_table(rows) -> dict[tuple, float]:
    """Peak Mcells/s per (component, mode, grid, precision, backend)."""
    out = {}
    for (comp, mode), curves in _group(rows).items():
        for (grid, prec, backend), pts in curves.items():
            out[(comp, mode, grid, prec, backend)] = max(p[1] for p in pts)
    return out


def make_figure(rows, out_dir: Path) -> Path:
    g = _group(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (comp, mode) in zip(axes.flat, PANELS):
        curves = g.get((comp, mode), {})
        for (grid, prec, backend), pts in sorted(curves.items()):
            pts = sorted(set(pts))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=BACK_MK.get(backend, "x"),
                    color=GRID_COLOR.get(grid, "#777777"),
                    linestyle=PREC_LS.get(prec, ":"),
                    label=f"{grid} {prec[-2:]} {backend}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_title(f"{comp.upper()} — {mode} scaling")
        ax.set_xlabel("devices (MPI ranks / GPUs)")
        ax.set_ylabel("throughput (Mcells/s)")
        ax.grid(True, which="both", alpha=0.3)
        if curves:
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
    fig.suptitle("Throughput (Mcells/s) by grid — atmosphere + ocean, "
                 "f32 (dashed) + f64 (solid), CPU (o) / GPU (s)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "throughput_by_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/bcw_scaling/bcw_scaling_tidy.csv")
    p.add_argument("--out", default="docs/scaling")
    args = p.parse_args()
    rows = _read(Path(args.csv))
    out = make_figure(rows, Path(args.out))
    print(f"wrote {out}")
    print("PEAK Mcells/s per (component, mode, grid, precision, backend):")
    for k, v in sorted(peak_table(rows).items()):
        print(f"  {k[0]:5s} {k[1]:6s} {k[2]:13s} {k[3]:8s} {k[4]:3s}  "
              f"peak={v:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
