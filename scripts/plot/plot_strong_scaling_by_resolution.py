#!/usr/bin/env python
"""Per-resolution strong-scaling curves from a tidy scaling CSV.

The summary plotters (``plot_throughput_by_grid``, ``plot_scaling_efficiency``)
collapse resolution to a single peak per (grid, precision, backend).  This one
keeps EVERY resolution as its own curve — the view you need to see which
resolutions actually fill more devices under strong scaling (small resolutions
go comms-bound and flatten early; large ones keep scaling).

Reads the tidy CSV from ``aggregate_bcw_scaling.py`` and renders, for ONE grid:
  - top row:    the metric (SYPD by default) vs resource count, one line per
                resolution, log-log,
  - bottom row: parallel speedup vs the smallest resource count, against the
                ideal y=x line,
one column per precision.  Resource count = ``n_resource`` (GPUs for a GPU run,
cores for a CPU run), so a CPU and GPU run of the same grid both plot sensibly.

    python scripts/plot/plot_strong_scaling_by_resolution.py \\
        --csv results/.../cube_strong_tidy.csv --grid cubed-sphere \\
        --out results/.../plots
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

# Grids that mean the same thing under the two spellings used across the repo.
_GRID_ALIASES = {"cubed-sphere": "cubed-sphere", "cubed_sphere": "cubed-sphere"}
# Per-grid resolution label prefix (purely cosmetic for the legend).
_RES_PREFIX = {"cubed-sphere": "C", "latlon": "LL", "icosahedral": "I",
               "spectral": "T"}


def _read(csv_path: Path) -> list[dict]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def _canon_grid(g: str) -> str:
    return _GRID_ALIASES.get(g, g)


def _res_label(grid: str, res) -> str:
    return f"{_RES_PREFIX.get(_canon_grid(grid), '')}{res}"


def group_series(rows, grid: str, metric: str = "sypd"):
    """``precision -> resolution -> sorted [(n_resource, value)]`` for one grid.

    Strong mode only.  Drops rows with non-positive metric, missing resource
    count, or a different grid.  ``n_resource`` falls back to ``n_devices`` if
    the older column name is the only one present.
    """
    want = _canon_grid(grid)
    out: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if _canon_grid(r.get("grid", "")) != want:
            continue
        if r.get("mode") != "strong":
            continue
        try:
            n = int(r.get("n_resource") or r.get("n_devices"))
            v = float(r[metric])
        except (ValueError, KeyError, TypeError):
            continue
        if v <= 0 or n <= 0:
            continue
        out[r.get("precision", "")][r.get("resolution", "?")].append((n, v))
    # sort + dedup each curve
    return {
        prec: {res: sorted(set(pts)) for res, pts in by_res.items()}
        for prec, by_res in out.items()
    }


def compute_speedup(by_res: dict):
    """``resolution -> [(n, speedup)]`` normalised to each curve's MIN resource.

    Speedup(N) = metric(N) / metric(N0), N0 = smallest resource count measured
    for that resolution.  Ideal strong scaling is N/N0 (the y=x line when both
    axes are the resource ratio).
    """
    out = {}
    for res, pts in by_res.items():
        if not pts:
            continue
        n0, v0 = pts[0]  # already sorted ascending by n
        if v0 <= 0:
            continue
        out[res] = [(n, (v / v0)) for n, v in pts]
    return out


def make_figure(rows, grid: str, out_dir: Path, metric: str = "sypd") -> Path:
    # Fail LOUD on a typoed --grid/--metric instead of silently writing a blank
    # PNG (codex review): a missing metric column or a grid with no strong-mode
    # rows is a user error, not an empty result.
    if rows and metric not in rows[0]:
        raise SystemExit(
            f"metric {metric!r} is not a column in the CSV "
            f"(columns: {sorted(rows[0])})")
    series = group_series(rows, grid, metric)
    if not series:
        raise SystemExit(
            f"no strong-mode rows for grid={grid!r} metric={metric!r} — check "
            f"--grid/--metric (canonical grids: cubed-sphere, latlon, "
            f"icosahedral, spectral)")
    precisions = sorted(series) or [""]
    ncol = len(precisions)
    fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 9), squeeze=False)

    for j, prec in enumerate(precisions):
        by_res = series.get(prec, {})
        ax_v, ax_s = axes[0][j], axes[1][j]

        # --- top: absolute metric vs resources ---
        for res, pts in sorted(by_res.items(), key=lambda kv: _int_or_str(kv[0])):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax_v.plot(xs, ys, marker="o", label=_res_label(grid, res))
        _style(ax_v, f"{_canon_grid(grid)} strong — {prec or 'all'}",
               metric.upper(), logy=True)
        if by_res:
            ax_v.legend(fontsize=8)

        # --- bottom: speedup vs ideal ---
        speed = compute_speedup(by_res)
        all_n = sorted({n for pts in by_res.values() for n, _ in pts})
        for res, pts in sorted(speed.items(), key=lambda kv: _int_or_str(kv[0])):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax_s.plot(xs, ys, marker="o", label=_res_label(grid, res))
        if all_n:
            n0 = all_n[0]
            ax_s.plot(all_n, [n / n0 for n in all_n], "k--", lw=1,
                      label="ideal")
        _style(ax_s, f"speedup vs {prec or 'all'}", "speedup", logy=False)
        ax_s.set_xlabel("resources (GPUs / cores)")
        if speed:
            ax_s.legend(fontsize=8)

    fig.suptitle(
        f"Strong scaling by resolution — {_canon_grid(grid)} "
        f"(metric={metric})", y=1.0, fontsize=12)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"strong_by_resolution_{_canon_grid(grid)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _int_or_str(x):
    try:
        return (0, int(x))
    except (ValueError, TypeError):
        return (1, str(x))


def _style(ax, title, ylabel, logy: bool) -> None:
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Tidy CSV from aggregate_bcw_scaling.py.")
    p.add_argument("--grid", default="cubed-sphere",
                   help="Grid to plot (cubed-sphere | latlon | icosahedral | spectral).")
    p.add_argument("--metric", default="sypd",
                   help="Column to plot on the top row (sypd | mcells_per_s).")
    p.add_argument("--out", default="results/scaling",
                   help="Output directory for the PNG.")
    args = p.parse_args()
    rows = _read(Path(args.csv))
    out = make_figure(rows, args.grid, Path(args.out), args.metric)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
