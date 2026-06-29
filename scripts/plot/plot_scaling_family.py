#!/usr/bin/env python
"""Speedup-vs-ideal scaling figures, one per (component, mode).

The headline scaling deliverable: for atmosphere and ocean, weak AND strong, on
CPU (cores) and GPU (devices), across every grid type, at float32 and float64 --
the classic "scaling behaviour as a function of 2^n devices" figure a reviewer
expects.  The full request is a 6-axis matrix (component x mode x backend x grid
x precision x device-count) far too large for one panel, so it is sharded into a
FAMILY OF FOUR figures:

    scaling_atm_strong.png    scaling_atm_weak.png
    scaling_ocean_strong.png  scaling_ocean_weak.png

Each figure is a grid of panels: ROWS = backend (CPU, GPU), COLUMNS = grid type.
Each panel plots ONE line per precision (float32, float64):

    y = speedup  =  T(N0) / T(N)      (N0 = the smallest device count present)
    x = device count = 2^n           (cores for CPU, GPUs for GPU; log2 axis)

with an IDEAL reference that depends on the mode:
  * strong scaling -> ideal speedup = N / N0   (the diagonal: same problem, more
    devices should run proportionally faster);
  * weak scaling   -> ideal speedup = 1.0 (flat: work grows WITH the device
    count, so constant wall-time -- NOT the diagonal -- is perfect scaling).
The measured quantity T(N0)/T(N) is identical in both modes; only the ideal
reference line differs, which is exactly what distinguishes the two scaling
questions.

Not every (grid, backend) sweeps a long 2^n ladder -- spectral has no MPI path,
cubed-sphere caps at its 6 faces (3 on GPU), and lat-lon GPU is single-node
(<=4) until the multi-node work lands (issue #641).  Those panels are drawn
HONESTLY: the available point(s) are plotted and the panel is annotated with why
the ladder is short, never extended with a faked curve.

    python scripts/bench/aggregate_bcw_scaling.py --root $OUT --out $OUT/all_tidy.csv
    python scripts/plot/plot_scaling_family.py --csv $OUT/all_tidy.csv --out $OUT/plots
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
# Preferred left-to-right column order per component; grids not listed fall to
# the end in alphabetical order (data-driven, so a new grid still shows up).
_GRID_ORDER = {
    "atm": ["cubed-sphere", "latlon", "icosahedral", "spectral"],
    "ocean": ["latlon", "tripole", "mpas"],
}
_BACKENDS = ("CPU", "GPU")            # the two panel rows, always shown
# One colour + style per precision so a panel's two lines never collide.
_PREC_STYLE = {
    "float32": {"color": "tab:blue", "marker": "o", "ls": "-", "label": "f32"},
    "float64": {"color": "tab:red", "marker": "s", "ls": "--", "label": "f64"},
}
# Why a panel's ladder is short (annotation), keyed by (component, grid, backend).
# Only the genuinely-capped cells; everything else sweeps the full 2^n ladder.
_BLOCKED_NOTE = {
    ("atm", "spectral", "CPU"): "no MPI path\n(single device)",
    ("atm", "spectral", "GPU"): "no MPI path\n(single device)",
    ("atm", "cubed-sphere", "CPU"): "MPI caps at 6 faces",
    ("atm", "cubed-sphere", "GPU"): "face-scatter <=3 GPU",
    ("atm", "latlon", "GPU"): "single-node <=4 GPU (#641)",
}


def _canon_grid(g: str) -> str:
    return _GRID_ALIASES.get(g, str(g))


def _read(csv_path: Path) -> list[dict]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def _tree():
    """Arbitrarily-deep auto-nesting dict (avoids miscounting defaultdict levels)."""
    return defaultdict(_tree)


def group(rows):
    """Nest rows for plotting, keyed for the speedup figures.

    Returns ``component -> mode -> grid -> backend -> precision -> resolution ->
    sorted [(n_devices, time_per_step_ms)]``.  Drops rows with a non-positive or
    missing time / device-count / resolution, or an unknown backend.  Duplicate
    (everything, n) keeps the FASTEST (smallest time) measurement.
    """
    raw = _tree()
    for r in rows:
        component = str(r.get("component", "")).lower()
        backend = str(r.get("backend", "")).upper()
        grid = _canon_grid(r.get("grid", ""))
        mode = str(r.get("mode", "")).lower()
        precision = str(r.get("precision", "")).lower()
        if component not in _GRID_ORDER or backend not in _BACKENDS or not grid:
            continue
        if mode not in ("strong", "weak") or precision not in _PREC_STYLE:
            continue
        try:
            n = int(float(r.get("n_resource") or r.get("n_devices")))
            res = int(float(r.get("resolution")))
            t = float(r.get("time_per_step_ms"))
        except (TypeError, ValueError):
            continue
        if n <= 0 or res <= 0 or t <= 0:
            continue
        series = raw[component][mode][grid][backend][precision][res]
        # keep the fastest repeat for this device count
        if n not in series or t < series[n]:
            series[n] = t
    # freeze the innermost {n: t} dicts into sorted (n, t) lists
    out: dict = {}
    for comp, by_mode in raw.items():
        out[comp] = {}
        for mode, by_grid in by_mode.items():
            out[comp][mode] = {}
            for grid, by_back in by_grid.items():
                out[comp][mode][grid] = {}
                for back, by_prec in by_back.items():
                    out[comp][mode][grid][back] = {}
                    for prec, by_res in by_prec.items():
                        out[comp][mode][grid][back][prec] = {
                            res: sorted(s.items()) for res, s in by_res.items()}
    return out


def representative_resolution(by_prec_res) -> int | None:
    """Pick ONE resolution per panel: the longest ladder, tie-break largest res.

    ``by_prec_res`` is ``precision -> resolution -> [(n, t)]``.  We want a single
    line per precision, so the panel commits to one resolution -- the one whose
    union of device counts (across precisions) is largest (most informative
    curve); ties go to the finest (largest) resolution.
    """
    counts: dict[int, set] = defaultdict(set)
    for by_res in by_prec_res.values():
        for res, pts in by_res.items():
            counts[res].update(n for n, _ in pts)
    if not counts:
        return None
    return max(counts, key=lambda res: (len(counts[res]), res))


def speedup_curve(pts):
    """``[(n, time)] -> [(n, speedup)]`` with speedup(N) = T(N0)/T(N).

    N0 is the smallest device count in the series (its speedup is 1.0 by
    construction).  A single-point series returns just ``[(n0, 1.0)]``.
    """
    if not pts:
        return []
    _, t0 = pts[0]                      # pts sorted by n -> first is N0
    if t0 <= 0:
        return []
    return [(n, t0 / t) for n, t in pts if t > 0]


def ideal_curve(xs, mode: str):
    """Ideal speedup over device counts ``xs`` for ``mode``.

    strong -> N/N0 (diagonal); weak -> 1.0 (flat, work grows with N).
    """
    xs = sorted(xs)
    if not xs:
        return []
    x0 = xs[0]
    if mode == "weak":
        return [(x, 1.0) for x in xs]
    return [(x, x / x0) for x in xs]


def _panel(ax, by_prec_res, mode: str, note: str | None) -> bool:
    """Draw one (grid, backend) panel; return True if it plotted any data."""
    res = representative_resolution(by_prec_res)
    drew = False
    all_x: set = set()
    for prec, style in _PREC_STYLE.items():
        pts = by_prec_res.get(prec, {}).get(res, []) if res is not None else []
        sp = speedup_curve(pts)
        if not sp:
            continue
        ax.plot([n for n, _ in sp], [s for _, s in sp],
                marker=style["marker"], ls=style["ls"], color=style["color"],
                label=style["label"])
        all_x.update(n for n, _ in sp)
        drew = True
    if drew:
        ideal = ideal_curve(all_x, mode)
        ax.plot([x for x, _ in ideal], [y for _, y in ideal],
                ls=":", color="0.5", lw=1.2,
                label="ideal" + (" (=1, weak)" if mode == "weak" else ""))
        ax.set_xscale("log", base=2)   # device counts are powers of two
        ax.set_yscale("log", base=10)  # consistent base-10 y across our figures
        ax.legend(fontsize=7)
        if res is not None:
            ax.text(0.03, 0.97, f"res {res}", transform=ax.transAxes,
                    fontsize=7, va="top", ha="left", color="0.3")
    else:
        ax.text(0.5, 0.5, note or "no data", transform=ax.transAxes,
                fontsize=8, ha="center", va="center", color="0.5")
    if note and drew:
        ax.text(0.97, 0.03, note, transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha="right", color="0.45")
    return drew


def make_figures(rows, out_dir: Path) -> list[Path]:
    """One figure per (component, mode); rows=backend, cols=grid. Returns PNGs.

    Fail LOUD (``SystemExit``) on a CSV missing the columns we need or with no
    usable rows, instead of writing blank panels.
    """
    needed = {"component", "backend", "grid", "mode", "precision",
              "time_per_step_ms"}
    if rows and not needed.issubset(rows[0]):
        raise SystemExit(
            f"CSV missing columns {sorted(needed - set(rows[0]))} "
            f"(have: {sorted(rows[0])})")
    data = group(rows)
    if not data:
        raise SystemExit(
            "no usable rows (need component in {atm,ocean}, backend in "
            "{CPU,GPU}, mode in {strong,weak}, precision in {float32,float64}, "
            "and a positive time_per_step_ms / n_resource / resolution)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for component in sorted(data):
        for mode in sorted(data[component]):
            by_grid = data[component][mode]
            # column order: preferred grids first, then any extras alphabetically
            present = set(by_grid)
            ordered = [g for g in _GRID_ORDER[component] if g in present]
            ordered += sorted(present - set(ordered))
            ncol = max(len(ordered), 1)
            fig, axes = plt.subplots(
                len(_BACKENDS), ncol, squeeze=False,
                figsize=(3.4 * ncol + 0.5, 6.4))
            for col, grid in enumerate(ordered):
                for row, backend in enumerate(_BACKENDS):
                    ax = axes[row][col]
                    by_prec_res = by_grid.get(grid, {}).get(backend, {})
                    note = _BLOCKED_NOTE.get((component, grid, backend))
                    _panel(ax, by_prec_res, mode, note)
                    if row == 0:
                        ax.set_title(grid, fontsize=10)
                    if col == 0:
                        ax.set_ylabel(f"{backend}\nspeedup T(N0)/T(N)",
                                      fontsize=9)
                    if row == len(_BACKENDS) - 1:
                        unit = "cores" if backend == "CPU" else "GPUs"
                        ax.set_xlabel(f"devices (2^n {unit})", fontsize=9)
                    ax.grid(True, which="both", alpha=0.3)
            fig.suptitle(f"{component} {mode} scaling — speedup vs ideal "
                         f"(f32 / f64)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out_path = out_dir / f"scaling_{component}_{mode}.png"
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            written.append(out_path)
    return written


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path,
                   help="Tidy CSV from aggregate_bcw_scaling.py.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for the per-(component,mode) PNGs.")
    args = p.parse_args()
    written = make_figures(_read(args.csv), args.out)
    print(f"wrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
