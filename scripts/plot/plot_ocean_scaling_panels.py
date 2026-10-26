"""Two-panel ocean strong-scaling figure: wall time per step, and speed.

The layout follows the one the Oceananigans team publishes: one coloured
series per resolution, device count on a log x-axis, wall time per timestep
on the left and simulated years per day on the right, with a grey dashed
ideal anchored at each series' first measured point.

Every point is a measured receipt row.  Rows are keyed by (grid, backend,
precision, resolution, levels, device count) and the FASTEST receipt per key
wins, which makes this a "best measured" figure; a provenance CSV naming the
file behind each point is written next to it so any point can be traced.

Simulated years per day is taken from the receipt's own ``sypd`` field rather
than recomputed, so it always agrees with the timestep that run used, and the
timestep appears in the legend exactly as the reference figure does.  A
receipt without one is dropped rather than guessed at.

    python scripts/plot/plot_ocean_scaling_panels.py \
        --receipts /scratch/b/b381103/legoesm_scaling \
        --backend gpu --precision float64 --out fig_ocean_scaling.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Colours follow the reference figure's ordering by resolution, coarse first.
SERIES_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#000000",
                 "#9467bd", "#8c564b"]
SERIES_MARKERS = ["o", "s", "D", "*", "o", "^", "v"]

# An MPAS Voronoi mesh doubles its cell count per subdivision, so the nominal
# spacing halves.  Subdivision 9 is the 2.6-million-cell production mesh.
_MPAS_KM_AT_SUBDIV_9 = 15.0


def mpas_label(subdivision: int) -> str:
    km = _MPAS_KM_AT_SUBDIV_9 * 2.0 ** (9 - int(subdivision))
    return f"subdiv-{int(subdivision)} (~{km:g} km)"


def load(receipt_roots):
    """Every valid ocean receipt under the given roots, newest-wins per key."""
    best = {}
    for root in receipt_roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path).read()
                except OSError:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    meta = row.get("metadata") or {}
                    is_ocean = (meta.get("component") == "ocean"
                                or row.get("component") in
                                ("mpas_ocean", "ocean", "latlon_ocean"))
                    if not is_ocean or not row.get("valid"):
                        continue
                    ms = row.get("steady_median_ms")
                    nd = row.get("n_devices")
                    if not ms or not nd:
                        continue
                    key = (row.get("grid_type") or meta.get("grid"),
                           row.get("backend"), row.get("precision"),
                           row.get("mode"), row.get("resolution"),
                           row.get("nlev"), int(nd))
                    row["_file"] = path
                    if key not in best or ms < best[key]["steady_median_ms"]:
                        best[key] = row
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", nargs="+", required=True)
    ap.add_argument("--grid", default="mpas")
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--precision", default=None,
                    help="float32 / float64; default plots both as separate "
                         "series, because they are different experiments")
    ap.add_argument("--out", default="fig_ocean_scaling.pdf")
    args = ap.parse_args()

    best = load(args.receipts)

    # Strong ladders only: a weak series has a different meaning on these axes.
    series = defaultdict(dict)   # (resolution, precision) -> {ndev: row}
    for (grid, backend, precision, mode, resolution, _nlev, nd), row in best.items():
        if grid != args.grid or backend != args.backend or mode != "strong":
            continue
        if args.precision and precision != args.precision:
            continue
        series[(resolution, precision)][nd] = row

    if not series:
        raise SystemExit("no strong ocean receipts matched the selection")

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    prov = []
    for i, key in enumerate(sorted(series, key=lambda k: (k[0], k[1]))):
        resolution, precision = key
        points = series[key]
        ndevs = sorted(points)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        marker = SERIES_MARKERS[i % len(SERIES_MARKERS)]
        ms = [points[n]["steady_median_ms"] for n in ndevs]
        dts = {points[n].get("dt_seconds") or points[n].get("dt")
               for n in ndevs}
        dt = dts.pop() if len(dts) == 1 else None
        dt_txt = (f", dt {dt / 60:g} min" if dt and dt >= 60
                  else f", dt {dt:g} s" if dt else "")
        label = (mpas_label(resolution) if args.grid == "mpas"
                 else str(resolution))
        label = f"{label} {precision[-2:]}-bit{dt_txt}"

        ax_t.plot(ndevs, ms, marker=marker, color=color, lw=2.0,
                  markersize=7, markerfacecolor="white", label=label)
        ideal = [ms[0] * ndevs[0] / n for n in ndevs]
        ax_t.plot(ndevs, ideal, ls="--", color="grey", lw=1.0, zorder=0)

        sypd = [points[n].get("sypd") for n in ndevs]
        have = [(n, s) for n, s in zip(ndevs, sypd) if s]
        if have:
            xs = [n for n, _ in have]
            ys = [s for _, s in have]
            ax_s.plot(xs, ys, marker=marker, color=color, lw=2.0,
                      markersize=7, markerfacecolor="white", label=label)
            ax_s.plot(xs, [ys[0] * n / xs[0] for n in xs],
                      ls="--", color="grey", lw=1.0, zorder=0)

        for n in ndevs:
            row = points[n]
            prov.append({"resolution": resolution, "precision": precision,
                         "n_devices": n,
                         "steady_median_ms": row["steady_median_ms"],
                         "sypd": row.get("sypd"),
                         "dt_seconds": row.get("dt_seconds") or row.get("dt"),
                         "nlev": row.get("nlev"), "file": row["_file"]})

    for ax in (ax_t, ax_s):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(f"{args.backend.upper()}s")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ticks = sorted({p["n_devices"] for p in prov})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
    ax_t.set_ylabel("Wall time per timestep [ms]")
    ax_s.set_ylabel("Simulated years per day (SYPD)")
    ax_s.legend(fontsize=8, frameon=False, loc="upper left")
    ax_t.plot([], [], ls="--", color="grey", lw=1.0, label="ideal")
    ax_t.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Ocean strong scaling — every point a measured receipt, "
                 "grey dashed is ideal", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")

    csv_path = os.path.splitext(args.out)[0] + "_provenance.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0].keys()))
        w.writeheader()
        w.writerows(prov)
    print(f"wrote {args.out}\nwrote {png}\nwrote {csv_path}")
    print(f"{len(prov)} points across {len(series)} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
