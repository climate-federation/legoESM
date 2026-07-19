#!/usr/bin/env python
"""Paper figure: dycore throughput across grids and resolutions (A100).

A CAPABILITY figure, not a scaling figure: "how fast does this model process
cells, on which grid, at which resolution".

METRIC IS Mcells/s ONLY -- SYPD IS DELIBERATELY NOT PLOTTED.
SYPD = f(dt), and the route-B SPMD lanes (lat-lon, icosahedral GPU) do not
carry a physical dt: ``bench_atm_latlon_spmd_scaling.py`` defaults
``--dt 60.0`` and the recorded dt is 60 s at LL192, LL512 AND LL1024 -- a CFL
timestep must shrink as the grid refines, so a constant one is a placeholder.
The route-A lanes (all CPU rows, plus every cubed-sphere row) DO carry the
CFL dt from ``_auto_dt`` and reproduce it exactly. Mixing the two produced a
260x error in an earlier draft's headline SYPD, so this figure reports only
the timestep-free metric. See the campaign report's GAPS entry on dt
provenance before reinstating any SYPD panel.

Mcells/s is also what makes lat-lon and cubed-sphere comparable at all: they
are indistinguishable per device (~1010 vs ~1030 Mcells/s on one A100).

CAVEAT the figure cannot remove: "matched resolution" across grids is not
matched WORK -- at ~40-56 km LL512 carries 13.6M cells, C192 5.8M, L7 4.3M.
Read these as per-grid capability curves, NOT as a grid ranking.

Colour = grid (categorical, fixed order, never cycled). The palette is a
validated triple (all six checks pass, no CVD warning) and is deliberately
NOT the Blues ramp of the scaling figures, where colour means resolution.

Run:  python scripts/plot/plot_grid_throughput.py \
          --csv .../all_tidy.csv [--out results/paper_figs]
"""
from __future__ import annotations

import argparse
import csv
import os

# Validated categorical triple (blue / orange / purple), fixed order.
GRID_STYLE = [
    ("latlon", "lat-lon", "#2c6fbb", "o", "-"),
    ("cubed-sphere", "cubed-sphere", "#b5562a", "^", "--"),
    ("icosahedral", "icosahedral", "#7a5aa8", "D", "-."),
]
INK, MUTED = "#1a1a1a", "#666666"


def load(csv_path):
    """GPU rows -> {grid: {resolution: {n_devices: (sypd, mcells, km)}}}."""
    if not csv_path:
        raise SystemExit("--csv is required: path to all_tidy.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"--csv not found: {csv_path}")
    out: dict = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["backend"] != "GPU":
                continue
            (out.setdefault(r["grid"], {})
                .setdefault(int(r["resolution"]), {})[int(r["n_devices"])]) = (
                    float(r["sypd"]), float(r["mcells_per_s"]),
                    float(r["resolution_km"]))
    return out


def series(grid_data, metric_idx, device="one"):
    """(km, value) pairs for one grid, sorted fine->coarse by km.

    ``device``: "one" = the 1-GPU point; "best" = the best value over the
    whole device ladder (reported WITH its N by the caller, since several
    configurations peak below the top of the ladder).
    """
    pts = []
    for _res, by_n in grid_data.items():
        if not by_n:
            continue
        km = next(iter(by_n.values()))[2]
        if device == "one":
            if 1 not in by_n:
                continue
            pts.append((km, by_n[1][metric_idx], 1))
        else:
            best_n = max(by_n, key=lambda n: by_n[n][metric_idx])
            pts.append((km, by_n[best_n][metric_idx], best_n))
    return sorted(pts)


def _style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="both", color="#ececec", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, color="#999999")


MCELLS, SYPD = 1, 0  # column indices into the (sypd, mcells, km) tuple


def make_figure(data, out, name="fig1_grid_throughput", show_best=True):
    import matplotlib.pyplot as plt

    fig, ax_only = plt.subplots(1, 1, figsize=(5.4, 4.2))

    # Single panel, Mcells/s only. A SYPD panel is intentionally absent --
    # see the module docstring (route-B dt is a placeholder).
    for metric_idx, ax, ylabel, title in (
            (MCELLS, ax_only, "throughput (Mcells / s)",
             "throughput per device"),):
        for key, label, col, mk, ls in GRID_STYLE:
            gd = data.get(key)
            if not gd:
                continue
            one = series(gd, metric_idx, "one")
            if one:
                ax.plot([p[0] for p in one], [p[1] for p in one], ls,
                        color=col, marker=mk, markersize=6.5, linewidth=2.0,
                        label=label, zorder=5, markeredgecolor="white",
                        markeredgewidth=0.7)
            if show_best:
                best = series(gd, metric_idx, "best")
                if best:
                    ax.plot([p[0] for p in best], [p[1] for p in best], ":",
                            color=col, marker=mk, markersize=4.5,
                            linewidth=1.3, alpha=0.55, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()  # fine resolution to the RIGHT reads as "harder"
        ax.set_xlabel("resolution (km)", fontsize=10, color=INK)
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        _style(ax)

    ax_only.legend(fontsize=8.5, frameon=False, loc="best")
    if show_best:
        ax_only.annotate("solid = 1 A100   ·   dotted = best over the ladder",
                         xy=(0.5, -0.24), xycoords="axes fraction",
                         fontsize=8, color=MUTED, ha="center")

    fig.suptitle("Dycore throughput by grid — dry dynamics, A100 (Derecho)",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = os.path.join(out, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.environ.get("WB_CAMPAIGN_CSV", ""))
    p.add_argument("--out", default="results/paper_figs")
    p.add_argument("--name", default="fig1_grid_throughput")
    p.add_argument("--no-best", action="store_true",
                   help="plot only the 1-GPU curves")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    data = load(a.csv)
    written = make_figure(data, a.out, a.name, show_best=not a.no_best)
    for path in written:
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
