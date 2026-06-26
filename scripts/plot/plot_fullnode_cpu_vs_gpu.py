#!/usr/bin/env python
"""Full-node CPU vs 1-A100 throughput, per grid, across resolutions.

The companion plot to the Derecho full-node comparison
(``scripts/cluster/scaling_derecho/submit_fullnode.sh``): each job writes one
full-node-CPU point and one 1-A100 point per resolution, so the meaningful view
is throughput vs RESOLUTION with one line per backend -- NOT throughput vs
device count (there is a single device config per side).  This complements
``plot_strong_scaling_by_resolution.py`` (SYPD vs resource count) and
``plot_throughput_by_grid.py`` (throughput vs devices).

Reads the tidy CSV from ``aggregate_bcw_scaling.py`` (run it over the whole
submit_fullnode outdir first) and renders, for EACH grid, a two-panel figure:
  - left:  SYPD vs resolution, one line per backend (CPU full node / GPU 1 A100)
  - right: throughput (Mcells/s) vs resolution, same lines,
with a printed CPU-vs-GPU speedup table per (grid, resolution).

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
# CPU = full node, GPU = 1 A100.  Colour + marker per backend.
_BACKEND_STYLE = {"CPU": ("#1f77b4", "o", "CPU (full node)"),
                  "GPU": ("#d62728", "s", "GPU (1 A100)")}
# Plotted metrics: column -> (axis label, log-y?).
_METRICS = (("sypd", "SYPD (sim-years/day)", True),
            ("mcells_per_s", "throughput (Mcells/s)", True))


def _canon_grid(g: str) -> str:
    return _GRID_ALIASES.get(g, g)


def _read(csv_path: Path) -> list[dict]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def _res_key(r) -> float:
    try:
        return float(r)
    except (TypeError, ValueError):
        return float("inf")


def group_by_grid(rows, metric: str = "sypd"):
    """``grid -> backend -> sorted [(resolution, value)]`` for one metric.

    Keeps the cube full-node hybrid and the per-resolution MPI points alike:
    one point per (grid, backend, resolution).  Drops rows with a non-positive
    or missing metric / resolution.  Backends are normalised to CPU / GPU.
    """
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grid = _canon_grid(r.get("grid", ""))
        backend = str(r.get("backend", "")).upper()
        if not grid or backend not in _BACKEND_STYLE:
            continue
        res = r.get("resolution")
        try:
            res_i = int(float(res))
            v = float(r[metric])
        except (TypeError, ValueError, KeyError):
            continue
        if v <= 0 or res_i <= 0:
            continue
        out[grid][backend].append((res_i, v))
    # sort + dedup (keep the best (max) value at a duplicated resolution)
    cleaned: dict[str, dict[str, list]] = {}
    for grid, by_back in out.items():
        cleaned[grid] = {}
        for backend, pts in by_back.items():
            best: dict[int, float] = {}
            for res_i, v in pts:
                best[res_i] = max(v, best.get(res_i, v))
            cleaned[grid][backend] = sorted(best.items())
    return cleaned


def speedup_table(rows):
    """``(grid, resolution) -> {cpu, gpu, gpu_over_cpu}`` from the SYPD points."""
    by_grid = group_by_grid(rows, "sypd")
    out = {}
    for grid, by_back in by_grid.items():
        cpu = dict(by_back.get("CPU", []))
        gpu = dict(by_back.get("GPU", []))
        for res in sorted(set(cpu) | set(gpu)):
            c, g = cpu.get(res), gpu.get(res)
            ratio = (g / c) if (c and g and c > 0) else None
            out[(grid, res)] = {"cpu": c, "gpu": g, "gpu_over_cpu": ratio}
    return out


def make_figures(rows, out_dir: Path) -> list[Path]:
    """One two-panel figure per grid; returns the written PNG paths.

    Fail LOUD (``SystemExit``) on an empty/typoed CSV instead of writing blank
    PNGs -- a CSV with no CPU/GPU rows for any grid is a user error.
    """
    if rows and not any(m in rows[0] for m, _, _ in _METRICS):
        raise SystemExit(
            f"CSV has none of the metric columns {[m for m, _, _ in _METRICS]} "
            f"(columns: {sorted(rows[0])})")
    by_grid_sypd = group_by_grid(rows, "sypd")
    if not by_grid_sypd:
        raise SystemExit(
            "no CPU/GPU rows with a positive SYPD found — check the CSV / "
            "--csv path (expected backends CPU and GPU, a 'grid' and "
            "'resolution' column)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for grid in sorted(by_grid_sypd):
        fig, axes = plt.subplots(1, len(_METRICS),
                                 figsize=(6 * len(_METRICS), 5), squeeze=False)
        for ax, (metric, ylabel, logy) in zip(axes[0], _METRICS):
            by_back = group_by_grid(rows, metric).get(grid, {})
            for backend, pts in sorted(by_back.items()):
                if not pts:
                    continue
                color, marker, label = _BACKEND_STYLE[backend]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker=marker, color=color, label=label)
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log", base=2)
            ax.set_xlabel(f"{grid} resolution")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{grid} — {metric}")
            ax.grid(True, which="both", alpha=0.3)
            if by_back:
                ax.legend(fontsize=9)
        fig.suptitle(f"Full-node CPU vs 1 A100 — {grid}")
        fig.tight_layout()
        out_path = out_dir / f"fullnode_cpu_vs_gpu_{grid}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        written.append(out_path)
    return written


def _print_speedup(rows) -> None:
    tbl = speedup_table(rows)
    if not tbl:
        return
    print(f"\n{'grid':<14} {'res':>6} {'CPU sypd':>10} {'GPU sypd':>10} "
          f"{'GPU/CPU':>8}")
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
    _print_speedup(rows)
    print(f"\nwrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
