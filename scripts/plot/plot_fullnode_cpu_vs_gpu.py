#!/usr/bin/env python
"""CPU vs GPU strong scaling, per grid: throughput (top) + efficiency (bottom).

The companion plot to the Derecho CPU-vs-A100 comparison
(``scripts/cluster/scaling_derecho/submit_fullnode.sh``): each grid is swept on
BOTH backends across their parallel units -- CPU MPI ranks 1..128 (cores) and
GPU 1..4 A100 -- at each resolution.  CPU cores and GPUs are DIFFERENT units, so
the two backends get their own columns (never compare a CPU point to a GPU point
at the same x); the cross-backend headline is the peak-SYPD table, not the axes.

Per grid, one 2x2 figure:
  - top-left   CPU: SYPD vs cores,  one line per resolution
  - top-right  GPU: SYPD vs A100,   one line per resolution
  - bottom-left  CPU parallel efficiency vs cores (ideal = 1)
  - bottom-right GPU parallel efficiency vs A100 (ideal = 1)
Efficiency(N) = SYPD(N) / (N/N0 * SYPD(N0)) relative to each curve's smallest
device count N0 -- it divides out ideal linear speedup, so the droop shows where
a backend goes comms-bound.  Also prints a peak (full-node) GPU/CPU table.

    python scripts/bench/aggregate_bcw_scaling.py --root $OUT --out $OUT/all_tidy.csv
    python scripts/plot/plot_fullnode_cpu_vs_gpu.py --csv $OUT/all_tidy.csv --out $OUT/plots
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Grid spellings that mean the same thing across the repo.
_GRID_ALIASES = {"cubed_sphere": "cubed-sphere"}
# The two columns: backend -> x-axis label.  CPU = cores, GPU = A100 count.
_BACKENDS = (("CPU", "MPI ranks (cores)"), ("GPU", "A100 GPUs"))


def _canon_grid(g: str) -> str:
    return _GRID_ALIASES.get(g, g)


def _read(csv_path: Path) -> list[dict]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def group(rows, metric: str = "sypd"):
    """``grid -> resolution -> backend -> sorted [(n_resource, value)]``.

    One strong-scaling curve per (grid, resolution, backend).  ``n_resource``
    is CPU cores or GPU count (from the aggregate).  Drops rows with a
    non-positive/missing metric, resource, or resolution; backends normalised
    to CPU / GPU.
    """
    known = {b for b, _ in _BACKENDS}
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        grid = _canon_grid(r.get("grid", ""))
        backend = str(r.get("backend", "")).upper()
        if not grid or backend not in known:
            continue
        try:
            res = int(float(r.get("resolution")))
            n = int(float(r.get("n_resource") or r.get("n_devices")))
            v = float(r[metric])
        except (TypeError, ValueError, KeyError):
            continue
        if v <= 0 or n <= 0 or res <= 0:
            continue
        out[grid][res][backend].append((n, v))
    clean: dict = {}
    for grid, by_res in out.items():
        clean[grid] = {}
        for res, by_back in by_res.items():
            clean[grid][res] = {}
            for backend, pts in by_back.items():
                best: dict[int, float] = {}
                for n, v in pts:
                    best[n] = max(v, best.get(n, v))
                clean[grid][res][backend] = sorted(best.items())
    return clean


def efficiency_curve(pts):
    """``[(n, sypd)] -> [(n, efficiency)]`` vs the smallest device count N0.

    efficiency(N) = (SYPD(N)/SYPD(N0)) / (N/N0); ideal linear scaling = 1.0.
    """
    if not pts:
        return []
    n0, v0 = pts[0]
    if v0 <= 0 or n0 <= 0:
        return []
    return [(n, (v / v0) / (n / n0)) for n, v in pts if n > 0]


def peak_table(rows):
    """``(grid, resolution) -> {cpu, gpu, gpu_over_cpu}`` peak (full-node) SYPD."""
    g = group(rows, "sypd")
    out = {}
    for grid, by_res in g.items():
        for res, by_back in by_res.items():
            cpu = max((v for _, v in by_back.get("CPU", [])), default=None)
            gpu = max((v for _, v in by_back.get("GPU", [])), default=None)
            ratio = (gpu / cpu) if (cpu and gpu) else None
            out[(grid, res)] = {"cpu": cpu, "gpu": gpu, "gpu_over_cpu": ratio}
    return out


def make_figures(rows, out_dir: Path) -> list[Path]:
    """One 2x2 figure per grid (SYPD top, efficiency bottom); returns the PNGs.

    Fail LOUD (``SystemExit``) on an empty/typoed CSV instead of blank PNGs.
    """
    if rows and "sypd" not in rows[0]:
        raise SystemExit(
            f"CSV has no 'sypd' column (columns: {sorted(rows[0])})")
    by_grid = group(rows, "sypd")
    if not by_grid:
        raise SystemExit(
            "no CPU/GPU rows with a positive SYPD found — check the CSV / "
            "--csv path (expected backends CPU and GPU and grid/resolution/"
            "n_resource columns)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    written = []
    for grid in sorted(by_grid):
        by_res = by_grid[grid]
        resolutions = sorted(by_res)
        # one colour per resolution, consistent across all four panels
        ncol = max(len(resolutions), 1)
        color = {res: cmap((i + 0.5) / ncol) for i, res in enumerate(resolutions)}

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
        max_eff = 1.0
        for col, (backend, xlabel) in enumerate(_BACKENDS):
            ax_top, ax_bot = axes[0][col], axes[1][col]
            for res in resolutions:
                pts = by_res[res].get(backend, [])
                if not pts:
                    continue
                ax_top.plot([n for n, _ in pts], [v for _, v in pts],
                            marker="o", color=color[res], label=f"res {res}")
                eff = efficiency_curve(pts)
                if eff:
                    ax_bot.plot([n for n, _ in eff], [e for _, e in eff],
                                marker="o", color=color[res])
                    max_eff = max(max_eff, max(e for _, e in eff))
            # top: absolute throughput
            ax_top.set_xscale("log", base=2)
            ax_top.set_yscale("log", base=2)
            ax_top.set_xlabel(xlabel)
            ax_top.set_ylabel("SYPD (sim-years/day)")
            ax_top.set_title(f"{backend} — throughput")
            ax_top.grid(True, which="both", alpha=0.3)
            if any(by_res[res].get(backend) for res in resolutions):
                ax_top.legend(fontsize=8, title="resolution")
            # bottom: parallel efficiency vs ideal=1
            ax_bot.axhline(1.0, ls="--", color="0.5", lw=1, label="ideal")
            ax_bot.set_xscale("log", base=2)
            ax_bot.set_xlabel(xlabel)
            ax_bot.set_ylabel("parallel efficiency")
            ax_bot.set_title(f"{backend} — efficiency (vs smallest count)")
            ax_bot.set_ylim(0, max(1.1, max_eff * 1.05))
            ax_bot.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"CPU vs GPU strong scaling — {grid}")
        fig.tight_layout()
        out_path = out_dir / f"fullnode_cpu_vs_gpu_{grid}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        written.append(out_path)
    return written


def _print_peak(rows) -> None:
    tbl = peak_table(rows)
    if not tbl:
        return
    print(f"\n{'grid':<14} {'res':>6} {'CPU peak':>10} {'GPU peak':>10} "
          f"{'GPU/CPU':>8}   (peak SYPD: full CPU node vs full GPU node)")
    for (grid, res) in sorted(tbl):
        d = tbl[(grid, res)]
        c = f"{d['cpu']:.3g}" if d["cpu"] is not None else "-"
        g = f"{d['gpu']:.3g}" if d["gpu"] is not None else "-"
        rr = f"{d['gpu_over_cpu']:.2f}x" if d["gpu_over_cpu"] is not None else "-"
        print(f"{grid:<14} {res:>6} {c:>10} {g:>10} {rr:>8}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path,
                   help="Tidy CSV from aggregate_bcw_scaling.py (over the "
                        "submit_fullnode outdir).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for the per-grid PNGs.")
    args = p.parse_args()
    rows = _read(args.csv)
    written = make_figures(rows, args.out)
    _print_peak(rows)
    print(f"\nwrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
