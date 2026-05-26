"""Plot single-GPU scaling from run_levante_gpu_scaling + bench_ocean_gpu_scaling CSVs.

Two figures:
  scaling_gpu_strong.png — wall-time per step vs total cells, log–log
  scaling_gpu_weak.png   — throughput (Mcells/s) vs total cells

Theoretical memory-bound floor overlaid. Reads any CSV produced by
the canonical TimingResult dataclass (atm or ocean).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_gpu_scaling.py \
        --atm results/scaling_gpu_baseline*/strong_scaling.csv \
        --ocean results/scaling_gpu_ocean/ocean_scaling.csv \
        --out results/scaling_gpu/
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Default = mobile RTX 5090 sustained measured ~730 GB/s (iter-8 probe).
# Desktop 5090 = ~1.79 TB/s peak (override via --peak-bw if benchmarked
# on a different device).
PEAK_BW_BYTES_S = 7.3e11

# Lower bound on bytes/cell-step (1 R + 1 W of 10 fp64 prognostic fields, no
# intermediates). Real dycores do RK3 substeps, hyperdiff passes, halo packs,
# tracer transport → 5-20× this floor. The 160 B figure is the optimistic
# roof — see scaling_gpu.md for measured effective bandwidth.
BYTES_PER_CELL_STEP = 10 * 2 * 8  # 160 B fp64


REQUIRED_COLS = ("total_cells", "time_per_step_ms", "mcells_per_s")


def _read_csv(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    matched_any = False
    for pat in paths:
        matches = glob.glob(pat)
        if not matches:
            print(f"WARN: no files match pattern {pat!r}")
            continue
        matched_any = True
        for p in matches:
            with open(p, newline="") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames is None:
                    print(f"WARN: {p} is empty — skipping")
                    continue
                missing = [c for c in REQUIRED_COLS if c not in rd.fieldnames]
                if missing:
                    print(f"WARN: {p} missing columns {missing} — skipping")
                    continue
                for ix, r in enumerate(rd):
                    r["_src"] = p
                    r["_row"] = ix
                    # Reject empty/NaN/inf numeric cells.
                    skip = False
                    for c in REQUIRED_COLS:
                        v = r.get(c, "")
                        try:
                            f_ = float(v)
                            if not np.isfinite(f_):
                                skip = True
                                break
                        except (TypeError, ValueError):
                            skip = True
                            break
                    if skip:
                        print(f"WARN: {p} row {ix} has bad numerics — skipping")
                        continue
                    rows.append(r)
    if not matched_any and paths:
        print("ERROR: no input CSV globs matched any files")
    return rows


def _group(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r.get(key, "?"), []).append(r)
    return out


def _label_from_row(r: dict) -> str:
    """Use CSV metadata for label rather than parent-dir heuristic.

    Combines (mode/grid prefix, precision) so fp32 vs fp64 are distinct
    series. Backstop to parent-dir name when CSV metadata is incomplete.
    """
    src = r.get("_src", "")
    parent = Path(src).parent.name.lower() if src else ""
    prec = r.get("precision", "").strip()
    mode = r.get("mode", "").strip()
    res = r.get("resolution", "")

    # Grid family from parent dir keyword
    grid = "unknown"
    for kw in ("spectral", "cubed-sphere", "cubed_sphere",
               "icosahedral", "latlon", "ocean", "mpas"):
        if kw in parent:
            grid = kw.replace("_", "-")
            break
    # Ocean rows carry mode='ocean_strong' even if dir says "fp32"
    if mode.startswith("ocean"):
        if "mpas" in parent or "voronoi" in parent or str(res).startswith("I"):
            grid = "ocean-mpas"
        else:
            grid = "ocean-latlon"
    return f"{grid} ({prec or 'unk'})"


def _theoretical_floor(total_cells: np.ndarray) -> np.ndarray:
    """Memory-bound ms/step floor."""
    return total_cells * BYTES_PER_CELL_STEP / PEAK_BW_BYTES_S * 1000.0


def plot_throughput(rows: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        lbl = _label_from_row(r)
        groups.setdefault(lbl, []).append(r)
    cmap = plt.get_cmap("tab10")
    for i, (label, rs) in enumerate(sorted(groups.items())):
        xs = np.array([int(r["total_cells"]) for r in rs])
        ys = np.array([float(r["mcells_per_s"]) for r in rs])
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], "o-", label=label, color=cmap(i % 10))
    # roofline: peak throughput (Mcells/s) = BW / bytes_per_cell_step / 1e6
    peak_mcells = PEAK_BW_BYTES_S / BYTES_PER_CELL_STEP / 1e6
    ax.axhline(peak_mcells, ls="--", color="k",
               label=f"memory-bound roof ({peak_mcells:.0f} Mcells/s)")
    ax.set_xscale("log")
    ax.set_xlabel("total cells (n_h × n_lev)")
    ax.set_ylabel("Throughput  [Mcells / s]")
    ax.set_title("Single-GPU dycore throughput (RTX 5090)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_time_per_step(rows: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        lbl = _label_from_row(r)
        groups.setdefault(lbl, []).append(r)
    cmap = plt.get_cmap("tab10")
    all_cells: list[int] = []
    for i, (label, rs) in enumerate(sorted(groups.items())):
        xs = np.array([int(r["total_cells"]) for r in rs])
        ys = np.array([float(r["time_per_step_ms"]) for r in rs])
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], "o-", label=label, color=cmap(i % 10))
        all_cells.extend(xs.tolist())
    if all_cells:
        grid = np.logspace(np.log10(min(all_cells)),
                           np.log10(max(all_cells)), 60)
        ax.plot(grid, _theoretical_floor(grid), "k--",
                label="memory-bound floor (160 B/cell)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total cells (n_h × n_lev)")
    ax.set_ylabel("ms / step")
    ax.set_title("Single-GPU dycore wall-time vs problem size")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    # Dead-code reference removal — caught in adversarial review.


def plot_peak_bar(rows: list[dict], out: Path):
    """Bar chart: peak Mcells/s per (grid, precision).

    NOTE: `_label_from_row` must include grid+precision (verified above) so
    fp32 and fp64 rows yield separate bars. `total_cells` from the CSV
    contract is horizontal cells × levels (cell-levels), not horizontal
    cells alone — annotated as such.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        lbl = _label_from_row(r)
        groups.setdefault(lbl, []).append(r)

    # label, peak Mcells/s, total_cell_levels, resolution
    peak_data: list[tuple[str, float, int, str]] = []
    for label, rs in groups.items():
        best = max(rs, key=lambda r: float(r["mcells_per_s"]))
        peak_data.append((
            label, float(best["mcells_per_s"]),
            int(best["total_cells"]),
            str(best.get("resolution", "?")),
        ))
    peak_data.sort(key=lambda x: x[1], reverse=True)

    labels = [p[0] for p in peak_data]
    vals = [p[1] for p in peak_data]
    sizes = [p[2] for p in peak_data]
    resos = [p[3] for p in peak_data]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    bars = ax.bar(range(len(labels)), vals,
                  color=[cmap(i % 10) for i in range(len(labels))])
    for i, (b, v, n, r) in enumerate(zip(bars, vals, sizes, resos)):
        ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.01,
                f"{v:.0f}\n(res={r}, {n:,} cell·lev)",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Peak throughput at best resolution  [Mcells / s]")
    ax.set_title("Single-GPU peak throughput by grid × precision (mobile RTX 5090)")
    ax.grid(axis="y", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_weak(rows: list[dict], out: Path):
    """Per-cell wall time (ns/cell) vs problem size — flat = device-saturated."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        lbl = _label_from_row(r)
        groups.setdefault(lbl, []).append(r)
    cmap = plt.get_cmap("tab10")
    for i, (label, rs) in enumerate(sorted(groups.items())):
        xs = np.array([int(r["total_cells"]) for r in rs])
        ms = np.array([float(r["time_per_step_ms"]) for r in rs])
        # ns/cell = (ms/step * 1e6 ns/ms) / cells.  Flat curve ⇒ device saturated.
        ns_per_cell = ms * 1e6 / xs
        order = np.argsort(xs)
        ax.plot(xs[order], ns_per_cell[order], "o-",
                label=label, color=cmap(i % 10))
    # Memory-bound floor: ns/cell = BYTES_PER_CELL_STEP / PEAK_BW_BYTES_S * 1e9
    ns_floor = BYTES_PER_CELL_STEP / PEAK_BW_BYTES_S * 1e9
    ax.axhline(ns_floor, ls="--", color="k",
               label=f"memory-bound floor ({ns_floor:.3f} ns/cell)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total cells (n_h × n_lev)")
    ax.set_ylabel("wall time per cell  [ns / cell / step]")
    ax.set_title("Single-GPU per-cell cost (flat = saturated, ↓ = improving)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    global PEAK_BW_BYTES_S
    p = argparse.ArgumentParser()
    p.add_argument("--atm", nargs="*", default=[])
    p.add_argument("--ocean", nargs="*", default=[])
    p.add_argument("--out", default="results/scaling_gpu")
    p.add_argument("--peak-bw", type=float, default=PEAK_BW_BYTES_S,
                   help="Peak HBM bandwidth in bytes/s (default: mobile "
                        "RTX 5090 measured sustained = 7.3e11; desktop "
                        "5090 = 1.79e12)")
    args = p.parse_args()
    PEAK_BW_BYTES_S = args.peak_bw
    rows = _read_csv(args.atm + args.ocean)
    if not rows:
        print("no rows found")
        return 1
    out_dir = Path(args.out)
    plot_throughput(rows, out_dir / "scaling_gpu_throughput.png")
    plot_time_per_step(rows, out_dir / "scaling_gpu_strong.png")
    plot_weak(rows, out_dir / "scaling_gpu_weak.png")
    plot_peak_bar(rows, out_dir / "scaling_gpu_peak_bar.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
