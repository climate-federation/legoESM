#!/usr/bin/env python
"""CPU vs GPU strong-scaling curves, per grid, across resolutions.

The companion plot to the Derecho CPU-vs-A100 comparison
(``scripts/cluster/scaling_derecho/submit_fullnode.sh``): each grid is swept on
BOTH backends across their parallel units -- CPU MPI ranks 1..128 (cores) and
GPU 1..4 A100 -- at each resolution.  So the view is throughput vs device count
(a strong-scaling curve), one curve per backend, one panel per resolution, one
figure per grid.  Complements ``plot_strong_scaling_by_resolution.py`` (single
grid/backend) and ``plot_throughput_by_grid.py`` (throughput vs devices, all
grids on one axis).

Reads the tidy CSV from ``aggregate_bcw_scaling.py`` (run it over the whole
submit_fullnode outdir first).  Per grid renders a [metric x resolution] grid of
panels -- rows = SYPD then Mcells/s, columns = resolution -- each panel showing
the CPU and GPU curves vs ``n_resource`` (CPU cores / GPU count).  Prints a
peak-throughput (full-node) CPU-vs-GPU table per (grid, resolution).

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
# CPU = cores, GPU = A100 count.  Colour + marker per backend.
_BACKEND_STYLE = {"CPU": ("#1f77b4", "o", "CPU (MPI ranks)"),
                  "GPU": ("#d62728", "s", "GPU (A100)")}
# Plotted metrics (rows): column -> (axis label).
_METRICS = (("sypd", "SYPD (sim-years/day)"),
            ("mcells_per_s", "throughput (Mcells/s)"))


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
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        grid = _canon_grid(r.get("grid", ""))
        backend = str(r.get("backend", "")).upper()
        if not grid or backend not in _BACKEND_STYLE:
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
    # sort + dedup (keep best value at a duplicated resource count)
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
    """One [metric x resolution] figure per grid; returns the written PNGs.

    Fail LOUD (``SystemExit``) on an empty/typoed CSV instead of blank PNGs.
    """
    if rows and not any(m in rows[0] for m, _ in _METRICS):
        raise SystemExit(
            f"CSV has none of the metric columns {[m for m, _ in _METRICS]} "
            f"(columns: {sorted(rows[0])})")
    grouped = {m: group(rows, m) for m, _ in _METRICS}
    grids = sorted(grouped["sypd"])
    if not grids:
        raise SystemExit(
            "no CPU/GPU rows with a positive SYPD found — check the CSV / "
            "--csv path (expected backends CPU and GPU and grid/resolution/"
            "n_resource columns)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for grid in grids:
        resolutions = sorted(grouped["sypd"][grid])
        ncol = max(len(resolutions), 1)
        fig, axes = plt.subplots(len(_METRICS), ncol,
                                 figsize=(5 * ncol, 4.5 * len(_METRICS)),
                                 squeeze=False)
        for row, (metric, ylabel) in enumerate(_METRICS):
            by_res = grouped[metric].get(grid, {})
            for col, res in enumerate(resolutions):
                ax = axes[row][col]
                by_back = by_res.get(res, {})
                for backend, pts in sorted(by_back.items()):
                    if not pts:
                        continue
                    color, marker, label = _BACKEND_STYLE[backend]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker=marker, color=color, label=label)
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=2)
                ax.set_xlabel("n_resource (CPU cores / GPUs)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{grid} {metric} — res {res}")
                ax.grid(True, which="both", alpha=0.3)
                if by_back:
                    ax.legend(fontsize=8)
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
          f"{'GPU/CPU':>8}   (peak SYPD across the scaling curve)")
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
