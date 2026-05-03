#!/usr/bin/env python
"""Render iter-218 scaling plots from the quick-sweep CSV.

Reads ``results/scaling/iter218_throughput.csv`` (produced by
``scripts/run_scaling_iter218.sh``) and produces:

  results/scaling/scaling.png

A 4-panel figure:
  (a) Throughput per (grid, resolution) on CPU vs GPU
  (b) GPU speedup factor over CPU
  (c) Mcells/s normalised by problem size
  (d) Comparison vs the iter-217 reference numbers from scaling.md

The plot is the user-facing summary referenced from scaling.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----- iter-217 reference (from scaling.md TL;DR) -----------------------
# This is the "what we had before iter-218" baseline so the new chart
# panel (d) shows whether we regressed or improved.
ITER217_REF = {
    # (grid, resolution): {"cpu": sps, "gpu": sps}
    ("spectral",      "T21"): {"cpu": 38.2,  "gpu": 288.4},
    ("spectral",      "T42"): {"cpu":  4.9,  "gpu": 117.6},
    ("icosahedral",   "I4"):  {"cpu": 98.8,  "gpu": 205.5},
    ("icosahedral",   "I5"):  {"cpu": 19.1,  "gpu": 198.5},
    ("cubed-sphere",  "C48"): {"cpu": 20.2,  "gpu": 103.0},
    ("cubed-sphere",  "C96"): {"cpu":  None, "gpu":  67.0},
}


def _ncells(grid: str, res: str) -> int:
    """Total cell count for a (grid, resolution) pair."""
    res = res.upper()
    if grid == "spectral":
        # Gaussian grid: 3*n + 1 lats × 2*(3*n + 1) lons; n_max=21 → ~64×128 ≈ 8192
        n = int(res.lstrip("T"))
        n_lat = 3 * n + 1
        n_lon = 2 * n_lat
        return n_lat * n_lon
    if grid == "icosahedral":
        # nCells = 10 * 4^level + 2
        level = int(res.lstrip("I"))
        return 10 * 4 ** level + 2
    if grid == "cubed-sphere":
        # 6 faces × n × n
        n = int(res.lstrip("C"))
        return 6 * n * n
    return 0


def load_throughput(csv_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Load the iter-218 CSV into ``(grid, resolution) → {backend: sps}``."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            grid = row["grid"]
            res = row["resolution"]
            res_disp = res if not res.isdigit() else (
                f"I{res}" if grid == "icosahedral" else f"C{res}"
            )
            try:
                sps = float(row["steps_per_sec"]) if row["steps_per_sec"] else None
            except ValueError:
                sps = None
            key = (grid, res_disp)
            out.setdefault(key, {})[row["backend"]] = sps
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path,
                        default=Path("results/scaling/iter218_throughput.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/scaling/scaling.png"))
    args = parser.parse_args(argv)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    data = load_throughput(args.csv)
    keys = sorted(data.keys())
    if not keys:
        print(f"No data in {args.csv}")
        return 1

    grid_colors = {
        "spectral":     "#1f77b4",
        "icosahedral":  "#2ca02c",
        "cubed-sphere": "#d62728",
        "latlon":       "#9467bd",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("legoESM iter-218 BCW scaling sweep", fontsize=14)
    ax_thr, ax_spd = axes[0]
    ax_eff, ax_cmp = axes[1]

    labels = [f"{g}/{r}" for g, r in keys]
    cpu_sps = [data[k].get("cpu") or np.nan for k in keys]
    gpu_sps = [data[k].get("gpu") or np.nan for k in keys]
    bar_x = np.arange(len(keys))
    width = 0.38

    ax_thr.bar(bar_x - width / 2, cpu_sps, width, label="CPU",
                color="#666666")
    ax_thr.bar(bar_x + width / 2, gpu_sps, width, label="GPU (RTX 5090 Lap.)",
                color=[grid_colors.get(g, "#000000") for g, _ in keys])
    ax_thr.set_ylabel("Throughput [steps/s]")
    ax_thr.set_xticks(bar_x)
    ax_thr.set_xticklabels(labels, rotation=20, ha="right")
    ax_thr.set_title("(a) BCW throughput, 1 sim-day")
    ax_thr.set_yscale("log")
    ax_thr.legend()
    ax_thr.grid(True, axis="y", alpha=0.3, which="both")

    speedup = [
        (data[k].get("gpu") / data[k].get("cpu"))
        if (data[k].get("gpu") and data[k].get("cpu")) else np.nan
        for k in keys
    ]
    ax_spd.bar(bar_x, speedup, color=[grid_colors.get(g, "#000000")
                                       for g, _ in keys])
    ax_spd.axhline(1.0, color="gray", lw=1, ls="--")
    ax_spd.set_ylabel("GPU / CPU throughput")
    ax_spd.set_xticks(bar_x)
    ax_spd.set_xticklabels(labels, rotation=20, ha="right")
    ax_spd.set_title("(b) GPU speed-up factor")
    for i, s in enumerate(speedup):
        if not np.isnan(s):
            ax_spd.text(i, s, f"{s:.1f}×", ha="center", va="bottom",
                          fontsize=9)
    ax_spd.grid(True, axis="y", alpha=0.3)

    # Panel (c): Mcells/s/device — efficiency surrogate.
    eff_cpu, eff_gpu = [], []
    nlev = 26
    for g, r in keys:
        nc = _ncells(g, r)
        eff_cpu.append((nc * nlev * (data[(g, r)].get("cpu") or 0)) / 1e6)
        eff_gpu.append((nc * nlev * (data[(g, r)].get("gpu") or 0)) / 1e6)
    ax_eff.bar(bar_x - width / 2, eff_cpu, width, label="CPU",
                color="#666666")
    ax_eff.bar(bar_x + width / 2, eff_gpu, width, label="GPU",
                color=[grid_colors.get(g, "#000000") for g, _ in keys])
    ax_eff.set_ylabel("Mcells/s")
    ax_eff.set_xticks(bar_x)
    ax_eff.set_xticklabels(labels, rotation=20, ha="right")
    ax_eff.set_title("(c) Mcells/s (cell-level throughput)")
    ax_eff.legend()
    ax_eff.grid(True, axis="y", alpha=0.3)

    # Panel (d): regression vs iter-217 reference.
    ref_gpu = []
    cur_gpu = []
    cmp_labels = []
    for k in keys:
        ref_entry = ITER217_REF.get(k)
        if ref_entry is None:
            continue
        ref_gpu.append(ref_entry.get("gpu"))
        cur_gpu.append(data[k].get("gpu"))
        cmp_labels.append(f"{k[0]}/{k[1]}")
    if cmp_labels:
        x = np.arange(len(cmp_labels))
        ax_cmp.bar(x - width / 2, ref_gpu, width, label="iter-217 ref",
                    color="#999999")
        ax_cmp.bar(x + width / 2, cur_gpu, width, label="iter-218 (this)",
                    color="#1f77b4")
        ax_cmp.set_ylabel("GPU throughput [steps/s]")
        ax_cmp.set_xticks(x)
        ax_cmp.set_xticklabels(cmp_labels, rotation=20, ha="right")
        ax_cmp.set_title("(d) iter-217 reference vs iter-218 measurement")
        ax_cmp.legend()
        ax_cmp.grid(True, axis="y", alpha=0.3)
        for i, (r, c) in enumerate(zip(ref_gpu, cur_gpu)):
            if r and c:
                pct = (c - r) / r * 100
                ax_cmp.text(i, max(r, c) * 1.02,
                              f"{pct:+.0f}%",
                              ha="center", va="bottom", fontsize=9,
                              color="green" if pct >= 0 else "red")
    else:
        ax_cmp.text(0.5, 0.5, "no overlap with iter-217 ref",
                     ha="center", va="center")
        ax_cmp.set_title("(d) iter-217 vs iter-218 — N/A for quick run")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
