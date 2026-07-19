#!/usr/bin/env python
"""CPU vs GPU strong scaling, per grid: SYPD (top) + Mcells/s throughput (bottom).

The companion plot to the Derecho CPU-vs-A100 comparison
(``scripts/cluster/scaling_derecho/submit_scaling.sh``): each grid is swept on
BOTH backends across their parallel units -- CPU MPI ranks (shown as NODES, 128
cores/node, from a fraction of a node up to the multi-node sweep) and GPU A100
count -- at each resolution.  CPU nodes and GPUs are DIFFERENT units, so the two
backends get their own columns (never compare a CPU point to a GPU point at the
same x); the cross-backend headline is the peak-SYPD table, not the axes.  Axes
are labelled in hardware units (nodes / A100s), not base-2 exponents.

Per grid, one 2x2 figure:
  - top-left   CPU: SYPD vs nodes,        one line per resolution
  - top-right  GPU: SYPD vs A100,         one line per resolution
  - bottom-left  CPU: Mcells/s vs nodes,  one line per resolution
  - bottom-right GPU: Mcells/s vs A100,   one line per resolution
Mcells/s (cells x levels / step-time) is raw compute throughput; unlike SYPD it
normalises out the per-resolution timestep, so a plateau in the bottom row is
the bandwidth/comms wall (where adding parallel units stops buying throughput).
Also prints a peak GPU/CPU table.

    python scripts/bench/aggregate_bcw_scaling.py --root $OUT --out $OUT/all_tidy.csv
    python scripts/plot/plot_cpu_vs_gpu_scaling.py --csv $OUT/all_tidy.csv --out $OUT/plots
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Grid spellings that mean the same thing across the repo.
_GRID_ALIASES = {"cubed_sphere": "cubed-sphere"}
# Derecho hardware unit: an EPYC compute node is 128 cores; the CPU x-axis is
# expressed in NODES (cores / this), so a 16-node run reads "16", not "2048".
# Sub-node rank counts (<128 cores) render as fractions (1/128 ... 1/2 node).
_CPU_CORES_PER_NODE = 128
# The two columns: backend -> (x-axis label, divisor from n_resource to the
# display unit).  CPU: cores -> nodes (/128).  GPU: A100 count is already the
# unit (/1).  Axes are labelled in hardware units, not base-2 exponents.
_BACKENDS = (
    ("CPU", "CPU nodes (128 cores/node)", _CPU_CORES_PER_NODE),
    ("GPU", "A100 GPUs", 1),
)


def _hw_tick_label(v: float) -> str:
    """Tick label in hardware units: integer for >=1, unit-fraction below.

    GPU A100 counts and full-node CPU points are integers ("1", "2", ... "16");
    sub-node CPU points (cores < 128 -> nodes < 1) read as "1/128" ... "1/2".
    """
    if v >= 1:
        return f"{int(round(v))}"
    return f"1/{int(round(1.0 / v))}"


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
    known = {b for b, *_ in _BACKENDS}
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        grid = _canon_grid(r.get("grid", ""))
        backend = str(r.get("backend", "")).upper()
        if not grid or backend not in known:
            continue
        # Strong-scaling only: weak-scaling rows (mode='weak') share the same
        # grid/backend/resolution columns and would otherwise pollute the strong
        # curves (each weak point is a different problem size at a different
        # rank count).  A missing/blank mode is treated as strong so older CSVs
        # that predate the column still plot.
        if str(r.get("mode") or "strong").strip().lower() != "strong":
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


def res_km(rows) -> dict:
    """``(canon grid, resolution_int) -> resolution_km`` for legend labels.

    Pulled straight from the tidy CSV's ``resolution_km`` column so the curves
    are labelled by physical grid spacing (km) rather than the grid-native
    resolution index (cube face cells / ico level / lat count).
    """
    out: dict = {}
    for r in rows:
        grid = _canon_grid(r.get("grid", ""))
        try:
            res = int(float(r.get("resolution")))
            km = float(r.get("resolution_km"))
        except (TypeError, ValueError):
            continue
        if grid and res > 0 and km > 0:
            out[(grid, res)] = km
    return out


def _res_label(km_map: dict, grid: str, res: int) -> str:
    """Legend label: grid spacing in km, falling back to the raw index."""
    km = km_map.get((grid, res))
    return f"{km:.0f} km" if km else f"res {res}"


def peak_table(rows):
    """``(grid, resolution) -> {cpu, gpu, gpu_over_cpu}`` peak SYPD."""
    g = group(rows, "sypd")
    out = {}
    for grid, by_res in g.items():
        for res, by_back in by_res.items():
            cpu = max((v for _, v in by_back.get("CPU", [])), default=None)
            gpu = max((v for _, v in by_back.get("GPU", [])), default=None)
            ratio = (gpu / cpu) if (cpu and gpu) else None
            out[(grid, res)] = {"cpu": cpu, "gpu": gpu, "gpu_over_cpu": ratio}
    return out


# Paper layout: rows = backend, cols = grid. Colour = RESOLUTION on the same
# sequential Blues ramp the scaling figures use (coarse = light, fine = dark),
# so "colour means resolution" is consistent across the whole figure set.
_PAPER_GRIDS = ("cubed-sphere", "icosahedral", "latlon")
_RES_RAMP = ["#9ecae1", "#4292c6", "#084594"]
_PAPER_MARKERS = ["o", "^", "D"]
_PAPER_LINES = ["--", "-.", "-"]
# cubed-sphere is face-capped (1/2/3/6) so it cannot pack processes and runs
# 1 proc x 128 threads on CPU, leaning on XLA-CPU threading; lat-lon and
# icosahedral run 128 MPI procs/node. Same hardware, ~15x different per-node
# throughput -- so the cube CPU panel is NOT comparable to its neighbours.
_CUBE_CPU_NOTE = "1 proc × 128 threads\n(XLA-CPU, not MPI-packed)"


def make_paper_figure(rows, out_dir: Path, name: str = "fig_cpu_gpu_mcells",
                      metric: str = "mcells_per_s") -> list[Path]:
    """2x3 paper figure: CPU (top) / GPU (bottom) x cube / ico / lat-lon.

    Mcells/s ONLY -- SYPD is deliberately absent. SYPD = f(dt), and the
    route-B SPMD lanes (lat-lon + icosahedral GPU) carry a benchmark-default
    dt rather than a CFL one, so any SYPD from them is not physical. Mcells/s
    is timestep-free and valid across every lane.

    The y-axis is SHARED across all six panels: the CPU-vs-GPU comparison is
    read as the VERTICAL OFFSET between rows, which only works on one scale.

    Comparison caveat (see the campaign report SCOPE): for lat-lon and
    icosahedral the CPU rows are route A (mpi4jax, 2-D pencil) and the GPU
    rows are route B (jax.distributed/NCCL, 1-D lat-band), so transport,
    decomposition AND hardware all differ. Read this as AS-DEPLOYED
    throughput -- what each stack delivers as actually configured -- never as
    a hardware speed-up. Cubed-sphere alone is a controlled comparison (both
    lanes cs_spmd, same driver, same dt).
    """
    if rows and metric not in rows[0]:
        raise SystemExit(
            f"CSV has no {metric!r} column (columns: {sorted(rows[0])})")
    by_grid = group(rows, metric)
    km_map = res_km(rows)
    present = [g for g in _PAPER_GRIDS if g in by_grid]
    if not present:
        raise SystemExit(
            f"no rows for any of {_PAPER_GRIDS} — check the CSV/aggregator")

    fig, axes = plt.subplots(2, len(present), figsize=(3.5 * len(present), 6.4),
                             sharey=True, squeeze=False)

    for col, grid in enumerate(present):
        by_res = by_grid[grid]
        # Coarse -> fine so the ramp reads light -> dark with resolution.
        resolutions = sorted(by_res, key=lambda r: -km_map.get((grid, r), r))
        for row, (backend, xlabel, divisor) in enumerate(_BACKENDS):
            ax = axes[row][col]
            for i, res in enumerate(resolutions):
                pts = by_res[res].get(backend) or []
                if not pts:
                    continue
                xs = [n / divisor for n, _v in pts]
                ys = [v for _n, v in pts]
                ax.plot(xs, ys, _PAPER_LINES[i % len(_PAPER_LINES)],
                        color=_RES_RAMP[i % len(_RES_RAMP)],
                        marker=_PAPER_MARKERS[i % len(_PAPER_MARKERS)],
                        markersize=5.5, linewidth=1.8, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=_res_label(km_map, grid, res))
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _p: _hw_tick_label(v)))
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.grid(True, which="both", color="#ececec", linewidth=0.6,
                    zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8.5, color="#999999")
            if row == 0:
                ax.set_title(grid, fontsize=11, color="#1a1a1a")
            if row == len(_BACKENDS) - 1:
                ax.set_xlabel(xlabel, fontsize=9, color="#1a1a1a")
            else:
                ax.set_xlabel(xlabel, fontsize=9, color="#666666")
            if col == 0:
                ax.set_ylabel(f"{backend}\nthroughput (Mcells / s)",
                              fontsize=9.5, color="#1a1a1a")
            if backend == "CPU" and grid == "cubed-sphere":
                ax.annotate(_CUBE_CPU_NOTE, xy=(0.04, 0.04),
                            xycoords="axes fraction", fontsize=7,
                            color="#b5562a", va="bottom")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7.5, frameon=False, loc="best")

    fig.suptitle("Throughput, as deployed — dry dynamics (Derecho)\n"
                 "CPU = 128-core EPYC nodes · GPU = A100",
                 fontsize=11.5, color="#1a1a1a", y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def make_figures(rows, out_dir: Path) -> list[Path]:
    """One 2x2 figure per grid (SYPD top, Mcells/s throughput bottom); the PNGs.

    Fail LOUD (``SystemExit``) on an empty/typoed CSV instead of blank PNGs.
    """
    if rows and "sypd" not in rows[0]:
        raise SystemExit(
            f"CSV has no 'sypd' column (columns: {sorted(rows[0])})")
    if rows and "mcells_per_s" not in rows[0]:
        raise SystemExit(
            f"CSV has no 'mcells_per_s' column (columns: {sorted(rows[0])})")
    by_grid = group(rows, "sypd")
    by_grid_mc = group(rows, "mcells_per_s")
    km_map = res_km(rows)
    # Grids present under EITHER metric: route-B SPMD points are throughput-only
    # (no SYPD), so keying solely on SYPD would drop them entirely.
    all_grids = sorted(set(by_grid) | set(by_grid_mc))
    if not all_grids:
        raise SystemExit(
            "no CPU/GPU rows with a positive SYPD or Mcells/s found — check the "
            "CSV / --csv path (expected backends CPU and GPU and grid/resolution/"
            "n_resource columns)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    # (row index, metric grouping, y-axis label) for the two stacked rows.
    panels = (
        (0, by_grid, "SYPD (sim-years/day)"),
        (1, by_grid_mc, "throughput (Mcells/s)"),
    )
    written = []
    for grid in all_grids:
        # Resolutions from EITHER metric (throughput-only route-B has no SYPD).
        resolutions = sorted(set(by_grid.get(grid, {})) | set(by_grid_mc.get(grid, {})))
        # one colour per resolution, consistent across all four panels
        ncol = max(len(resolutions), 1)
        color = {res: cmap((i + 0.5) / ncol) for i, res in enumerate(resolutions)}

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
        for col, (backend, xlabel, divisor) in enumerate(_BACKENDS):
            for row, src, ylabel in panels:
                ax = axes[row][col]
                src_res = src.get(grid, {})
                panel_xs: set[float] = set()
                for res in resolutions:
                    pts = src_res.get(res, {}).get(backend, [])
                    if not pts:
                        continue
                    # n_resource -> hardware unit (CPU cores/128 = nodes; GPU=A100).
                    xs = [n / divisor for n, _ in pts]
                    panel_xs.update(xs)
                    ax.plot(xs, [v for _, v in pts],
                            marker="o", color=color[res],
                            label=_res_label(km_map, grid, res))
                # Powers-of-two spacing, but tick ONLY at the measured points and
                # label them in hardware units (nodes / A100s) -- no base-2
                # exponents and no log minor-tick clutter.
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=10)
                if panel_xs:
                    ax.set_xticks(sorted(panel_xs))
                    ax.xaxis.set_major_formatter(
                        FuncFormatter(lambda v, _pos=None: _hw_tick_label(v)))
                    ax.minorticks_off()
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{backend} — {ylabel.split(' (')[0]}")
                ax.grid(True, which="both", alpha=0.3)
                if any(src_res.get(res, {}).get(backend) for res in resolutions):
                    ax.legend(fontsize=8, title="grid spacing")
        fig.suptitle(f"CPU vs GPU strong scaling — {grid}")
        fig.tight_layout()
        out_path = out_dir / f"cpu_vs_gpu_scaling_{grid}.png"
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
                        "submit_scaling outdir).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for the per-grid PNGs.")
    p.add_argument("--layout", default="per-grid",
                   choices=["per-grid", "paper", "both"],
                   help="'per-grid' = one 2x2 diagnostic figure per grid "
                        "(SYPD + Mcells/s); 'paper' = ONE 2x3 figure, "
                        "CPU/GPU rows x grid columns, Mcells/s only "
                        "(SYPD omitted — route-B dt is a placeholder)")
    p.add_argument("--paper-name", default="fig_cpu_gpu_mcells")
    args = p.parse_args()
    rows = _read(args.csv)
    written = []
    if args.layout in ("per-grid", "both"):
        written += make_figures(rows, args.out)
    if args.layout in ("paper", "both"):
        written += make_paper_figure(rows, args.out, args.paper_name)
    if args.layout != "paper":
        _print_peak(rows)
    print(f"\nwrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
