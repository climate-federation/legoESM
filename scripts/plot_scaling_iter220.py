#!/usr/bin/env python
"""Render the iter-220 ``--scan-steps`` sweep curve and refresh scaling.png.

Reads ``results/scaling/iter220_scan_steps.csv`` (produced by
``scripts/run_scaling_iter220.sh``) and produces:

  results/scaling/scaling.png

A 2-panel figure:
  (a) GPU throughput vs scan-batch K, one curve per (grid, resolution).
      Annotated with the best-K-per-curve point.
  (b) Speed-up factor relative to K=1 — kernel-launch amortisation as
      a function of batch size.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _label(grid: str, res: str) -> str:
    res = res.upper()
    if grid == "icosahedral" and res.isdigit():
        res = f"I{res}"
    elif grid == "cubed-sphere" and res.isdigit():
        res = f"C{res}"
    return f"{grid}/{res}"


def load(csv_path: Path):
    """(grid, resolution) → sorted [(K, sps), ...]."""
    out: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            grid = row["grid"]
            res = row["resolution"]
            try:
                K = int(row["scan_steps"])
                sps = float(row["steps_per_sec"])
            except (ValueError, TypeError):
                continue
            out[(grid, res)].append((K, sps))
    for k, v in out.items():
        v.sort()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path,
                        default=Path("results/scaling/iter220_scan_steps.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/scaling/scaling.png"))
    args = parser.parse_args(argv)

    data = load(args.csv)
    if not data:
        print(f"No data in {args.csv}")
        return 1

    grid_colors = {
        "spectral":     "#1f77b4",
        "icosahedral":  "#2ca02c",
        "cubed-sphere": "#d62728",
    }
    grid_markers = {
        "spectral":     "o",
        "icosahedral":  "s",
        "cubed-sphere": "^",
    }

    fig, (ax_thr, ax_sp) = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle(
        "legoESM iter-220 — `--scan-steps` sweep on the BCW GPU benchmark",
        fontsize=13,
    )

    # The lines vary brightness by resolution to keep grid colour stable.
    res_brightness = {"T21": 1.0, "T42": 0.6,
                       "24":  1.0, "48":  0.6,
                       "4":   1.0, "5":   0.6}

    keys = sorted(data.keys())
    best_lookup = {}
    for (grid, res) in keys:
        pts = data[(grid, res)]
        K_arr = np.array([p[0] for p in pts])
        sps_arr = np.array([p[1] for p in pts])
        color_base = np.array(matplotlib.colors.to_rgb(grid_colors[grid]))
        color = tuple(color_base * res_brightness.get(res, 1.0)
                       + (1 - res_brightness.get(res, 1.0)) * np.array([1, 1, 1]))
        marker = grid_markers[grid]
        ax_thr.plot(K_arr, sps_arr, marker=marker, color=color,
                     label=_label(grid, res), linewidth=1.4)
        # Annotate best
        best_idx = int(np.argmax(sps_arr))
        best_K = int(K_arr[best_idx])
        best_sps = float(sps_arr[best_idx])
        best_lookup[(grid, res)] = (best_K, best_sps)
        ax_thr.scatter([best_K], [best_sps], s=55, edgecolor="black",
                        facecolor=color, zorder=4)
        ax_thr.text(best_K, best_sps * 1.02, f"K={best_K}\n{best_sps:.0f} sps",
                     fontsize=8, ha="center", color="black")

        # Speed-up vs K=1
        if K_arr[0] == 1:
            base = sps_arr[0]
            ax_sp.plot(K_arr, sps_arr / base, marker=marker, color=color,
                        label=_label(grid, res), linewidth=1.4)

    ax_thr.set_xscale("log", base=2)
    ax_thr.set_xticks([1, 6, 12, 24, 48])
    ax_thr.set_xticklabels([1, 6, 12, 24, 48])
    ax_thr.set_xlabel("--scan-steps K (log scale)")
    ax_thr.set_ylabel("GPU throughput [steps/s]")
    ax_thr.set_title("(a) Throughput vs scan-batch K")
    ax_thr.grid(True, alpha=0.3)
    ax_thr.legend(loc="upper left", ncol=2, fontsize=9)

    ax_sp.set_xscale("log", base=2)
    ax_sp.set_xticks([1, 6, 12, 24, 48])
    ax_sp.set_xticklabels([1, 6, 12, 24, 48])
    ax_sp.axhline(1.0, color="gray", lw=1, ls="--")
    ax_sp.set_xlabel("--scan-steps K (log scale)")
    ax_sp.set_ylabel("speed-up vs K=1")
    ax_sp.set_title("(b) Kernel-launch amortisation gain")
    ax_sp.grid(True, alpha=0.3)
    ax_sp.legend(loc="upper left", ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")

    # Print a concise best-K table
    print()
    print("Best K per (grid, resolution):")
    print(f"{'grid':>14} {'res':>5} {'K':>4} {'sps':>8} {'×K=1':>7}")
    for (grid, res), (K, sps) in sorted(best_lookup.items()):
        pts = data[(grid, res)]
        baseline = next((s for k, s in pts if k == 1), None)
        ratio = (sps / baseline) if baseline else float("nan")
        print(f"{grid:>14} {res:>5} {K:>4} {sps:8.1f} {ratio:6.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
