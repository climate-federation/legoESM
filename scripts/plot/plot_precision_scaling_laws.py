"""Strong-scaling laws across grids and precisions (f32/f64/mixed).

Reads the per-arm JSONL the strong-ladder sbatches emit
(``{grid}_{res}_d{ND}_x{0|1}.jsonl`` for atm; ocean arms carry precision in
the row). One curve per (grid, resolution, precision):

  * COLOUR encodes the GRID (same colour for all precisions of a grid).
  * LINE WIDTH encodes precision: f64 THIN, f32 THICK.
  * ocean ``mixed`` is DASHED (same colour, medium width).
  * marker shape encodes the resolution (coarse vs fine) within a grid.

Panels: (left) throughput [Mcells/s] vs GPUs, log-log, with the ideal linear
reference; (right) parallel efficiency vs GPUs (throughput-per-GPU normalised
to each curve's smallest-device point).

Usage:
  python scripts/plot/plot_precision_scaling_laws.py DIR [DIR ...] \
      --out results/scaling_plots/precision_scaling_laws.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# grid -> colour (stable, colour-blind-ish)
_GRID_COLOR = {
    "latlon": "#1f77b4",
    "mpas": "#d62728",
    "cube": "#2ca02c",
    "spectral": "#9467bd",
    "ocean-latlon": "#17becf",
    "ocean-mpas": "#ff7f0e",
}
# precision -> (linewidth, linestyle)
_PREC_STYLE = {
    "float32": (3.2, "-"),
    "float64": (1.3, "-"),
    "mixed": (2.2, "--"),
}
_RES_MARKER = {"a": "o", "b": "s", "0": "o", "1": "s"}

_FNAME = re.compile(r"(?P<grid>[a-z]+)_(?P<res>[a-z0-9]+)_d(?P<nd>\d+)_x(?P<x64>[01])")


def _last_json(path: Path):
    rec = None
    for line in path.open():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except Exception:
                pass
    return rec


def _throughput(rec) -> float | None:
    """Mcells/s, from the row or derived from step time + cell count."""
    for k in ("mcells_per_s", "mcells_s", "mcell_per_s"):
        if rec.get(k):
            return float(rec[k])
    ms = rec.get("steady_median_ms") or rec.get("time_per_step_ms")
    cells = rec.get("total_cells") or rec.get("cells") or rec.get("n_cells")
    if ms and cells:
        return float(cells) / (float(ms) * 1e-3) / 1e6
    return None


def _collect(dirs):
    """Return {(grid,res,prec): {nd: mcells_s}} keyed for plotting."""
    curves: dict = {}
    for d in dirs:
        for path in sorted(Path(d).glob("*.jsonl")):
            rec = _last_json(path)
            if not rec:
                continue
            m = _FNAME.search(path.stem)
            if m:
                grid = m.group("grid")
                res = m.group("res")
                nd = int(m.group("nd"))
                prec = "float64" if m.group("x64") == "1" else "float32"
            else:
                # ocean / other benches: read fields from the row.
                grid = rec.get("grid") or rec.get("grid_type") or path.stem.split("_")[0]
                if grid in ("latlon", "mpas") and "ocean" in str(path).lower():
                    grid = "ocean-" + grid
                res = str(rec.get("resolution") or rec.get("subdivision") or "a")
                nd = int(rec.get("n_devices") or rec.get("n_gpus") or 0)
                prec = rec.get("precision", "float64")
            tput = _throughput(rec)
            if not nd or tput is None:
                continue
            curves.setdefault((grid, res, prec), {})[nd] = tput
    return curves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="dirs of *.jsonl arms")
    ap.add_argument("--out", default="results/scaling_plots/precision_scaling_laws.png")
    ap.add_argument("--title", default="Strong-scaling laws (per-grid colour; f64 thin, f32 thick, mixed dashed)")
    args = ap.parse_args()

    curves = _collect(args.dirs)
    if not curves:
        print("no arms parsed from:", args.dirs)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    grids_seen = set()

    for (grid, res, prec), nd_tput in sorted(curves.items()):
        if len(nd_tput) < 2:
            continue
        color = _GRID_COLOR.get(grid, "#666666")
        lw, ls = _PREC_STYLE.get(prec, (2.0, "-"))
        marker = _RES_MARKER.get(res, "^")
        xs = sorted(nd_tput)
        ys = [nd_tput[x] for x in xs]
        grids_seen.add(grid)
        ax1.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, ms=6,
                 markerfacecolor="none")
        # ideal linear scaling from the smallest-device point of this curve.
        y0, x0 = ys[0], xs[0]
        ax1.plot(xs, [y0 * (x / x0) for x in xs], color=color, lw=0.6, ls=":",
                 alpha=0.35)
        # parallel efficiency: (tput/nd) normalised to the smallest-nd point.
        eff0 = ys[0] / x0
        eff = [(nd_tput[x] / x) / eff0 for x in xs]
        ax2.plot(xs, eff, color=color, lw=lw, ls=ls, marker=marker, ms=6,
                 markerfacecolor="none")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("GPUs")
        ax.grid(True, which="both", alpha=0.25)
    ax1.set_yscale("log")
    ax1.set_ylabel("throughput  [Mcells / s]")
    ax1.set_title("throughput vs GPUs (dotted = ideal linear)")
    ax2.axhline(1.0, color="black", ls=":", alpha=0.5)
    ax2.set_ylabel("parallel efficiency  (vs smallest-GPU point)")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("parallel efficiency")

    grid_handles = [Line2D([], [], color=_GRID_COLOR.get(g, "#666"), lw=3, label=g)
                    for g in sorted(grids_seen)]
    prec_handles = [
        Line2D([], [], color="#444", lw=3.2, ls="-", label="f32 (thick)"),
        Line2D([], [], color="#444", lw=1.3, ls="-", label="f64 (thin)"),
        Line2D([], [], color="#444", lw=2.2, ls="--", label="mixed (dashed)"),
    ]
    ax1.legend(handles=grid_handles, title="grid", loc="upper left", fontsize=9)
    ax2.legend(handles=prec_handles, title="precision", loc="lower left", fontsize=9)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(curves)} curves, grids={sorted(grids_seen)})")

    # Text receipt: the >=64-GPU efficiency per curve (the plateau number).
    print("\n=== >=64-GPU parallel efficiency (vs smallest-GPU point) ===")
    for (grid, res, prec), nd_tput in sorted(curves.items()):
        if len(nd_tput) < 2:
            continue
        xs = sorted(nd_tput)
        x0 = xs[0]
        eff0 = nd_tput[x0] / x0
        hi = [x for x in xs if x >= 64]
        for x in hi:
            e = (nd_tput[x] / x) / eff0
            print(f"  {grid:12s} {res:>4} {prec:8s} {x:>4} GPU  eff={e:5.1%}  "
                  f"tput={nd_tput[x]:8.1f} Mc/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
