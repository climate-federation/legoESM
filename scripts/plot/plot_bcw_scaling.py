"""Publication scaling figures for the baroclinic-wave campaign.

Reads the tidy CSV from ``aggregate_bcw_scaling.py`` and renders, for each
(case, precision) present, a two-panel figure in the style of the classic
SYPD-vs-device scaling plot:

    * top panel  = GPU,  bottom panel = CPU,
    * x = device count (log),  y = SYPD (log),
    * one COLOR per grid (icosahedral / latlon / cubed-sphere / spectral),
    * one solid measured curve per (grid, resolution); finer resolutions sit
      lower and are annotated with their nominal km spacing at the curve end,
    * a dotted IDEAL linear-strong-scaling reference for each curve
      (SYPD_ideal(n) = SYPD(n0) * n / n0 from the curve's smallest device
      count).

Strong-scaling rows only (the reference figure is strong scaling).  Pure
matplotlib + stdlib csv (no JAX, no pandas) so it runs on a login node.

Usage::

    python scripts/plot/plot_bcw_scaling.py \
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

# One color per grid (the user's "one color for each grid on the same plot").
GRID_COLOR = {
    "icosahedral": "#1f77b4",   # blue
    "latlon": "#ff7f0e",        # orange
    "cubed-sphere": "#2ca02c",  # green
    "cubed_sphere": "#2ca02c",
    "spectral": "#d62728",      # red
}
GRID_LABEL = {
    "icosahedral": "icosahedral (MPAS)",
    "latlon": "lat-lon (FV)",
    "cubed-sphere": "cubed-sphere (FV3)",
    "cubed_sphere": "cubed-sphere (FV3)",
    "spectral": "spectral",
}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
CASE_TITLE = {"dry": "Dry baroclinic wave", "moist": "Moist baroclinic wave"}


def _read(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("sypd", "resolution_km"):
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
        r["n_devices"] = int(r["n_devices"])
    return rows


def _panel(ax, rows, title):
    """Draw one backend panel; return True if anything was plotted."""
    # group: grid -> resolution -> list[(n_devices, sypd, km)]
    groups: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["grid"]][r["resolution"]].append(
            (r["n_devices"], r["sypd"], r["resolution_km"]))

    plotted = False
    for grid in sorted(groups, key=lambda g: GRID_LABEL.get(g, g)):
        color = GRID_COLOR.get(grid, "#555555")
        for mi, res in enumerate(sorted(groups[grid], key=lambda x: float(x))):
            pts = sorted(groups[grid][res])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            km = pts[0][2]
            mk = _MARKERS[mi % len(_MARKERS)]
            ax.plot(xs, ys, marker=mk, color=color, lw=1.8, ms=5, zorder=3)
            plotted = True
            # Dotted ideal linear strong scaling from the first point.
            if len(xs) >= 2:
                x0, y0 = xs[0], ys[0]
                xi = [xs[0], xs[-1]]
                yi = [y0, y0 * xs[-1] / x0]
                ax.plot(xi, yi, ls=":", color=color, lw=1.0, alpha=0.7, zorder=2)
            # km label at the curve end.
            if km == km:  # not NaN
                ax.annotate(f"{km:.0f} km", (xs[-1], ys[-1]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=7, color=color, va="center")

    if not plotted:
        return False
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylabel("SYPD")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title, fontsize=10, loc="left")
    # Grid legend (color key only; resolutions are annotated inline).
    handles = [plt.Line2D([], [], color=GRID_COLOR.get(g, "#555"), marker="o",
                          lw=1.8, label=GRID_LABEL.get(g, g))
               for g in sorted(groups, key=lambda g: GRID_LABEL.get(g, g))]
    handles.append(plt.Line2D([], [], color="0.4", ls=":", lw=1.0,
                              label="ideal (linear)"))
    ax.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)
    return True


def make_figure(rows, case, precision, out_dir: Path) -> Path | None:
    sub = [r for r in rows
           if r["case"] == case and r["precision"] == precision
           and r["mode"] == "strong" and r["backend"] in ("GPU", "CPU")
           and r["sypd"] == r["sypd"]]
    if not sub:
        return None
    fig, (ax_g, ax_c) = plt.subplots(2, 1, figsize=(7.0, 8.4), sharex=False)
    prec_lbl = {"float64": "FP64", "float32": "FP32"}.get(precision, precision)
    any_g = _panel(ax_g, [r for r in sub if r["backend"] == "GPU"],
                   f"GPU — {CASE_TITLE.get(case, case)} ({prec_lbl})")
    any_c = _panel(ax_c, [r for r in sub if r["backend"] == "CPU"],
                   f"CPU — {CASE_TITLE.get(case, case)} ({prec_lbl})")
    if not (any_g or any_c):
        plt.close(fig)
        return None
    ax_g.set_xlabel("Number of GPUs")
    ax_c.set_xlabel("Number of CPU cores")
    fig.tight_layout()
    out = out_dir / f"bcw_scaling_{case}_{precision}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/bcw_scaling/bcw_scaling_tidy.csv")
    p.add_argument("--out", default="docs/scaling")
    args = p.parse_args()

    rows = _read(Path(args.csv))
    out_dir = Path(args.out)
    made = []
    cases = sorted({r["case"] for r in rows})
    precs = sorted({r["precision"] for r in rows})
    for case in cases:
        for prec in precs:
            f = make_figure(rows, case, prec, out_dir)
            if f is not None:
                made.append(f)
    if not made:
        print("No strong-scaling rows to plot.")
        return 1
    for f in made:
        print(f"Wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
