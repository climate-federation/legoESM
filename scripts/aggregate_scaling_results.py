#!/usr/bin/env python
"""Aggregate CPU MPI scaling results and generate plots.

Reads JSON result files from ``results/cpu_scaling/`` and produces:
- Combined CSV
- Summary tables
- Weak and strong scaling plots per grid type

Usage::

    python scripts/aggregate_scaling_results.py results/cpu_scaling/
    python scripts/aggregate_scaling_results.py results/cpu_scaling/ --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


def load_results(base_dir: Path) -> list[dict]:
    """Recursively load all JSON result files."""
    results = []
    for p in sorted(base_dir.rglob("*.json")):
        if p.name == "metadata.json":
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if "time_per_step_ms" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            print(f"  Skipping {p}")
    return results


def compute_scaling_efficiency(results: list[dict]) -> list[dict]:
    """Compute scaling efficiency relative to fewest-rank baseline."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r["grid_type"], r["physics_level"], r["mode"],
               r["resolution"], r["precision"])
        groups.setdefault(key, []).append(r)

    for key, group in groups.items():
        _, _, mode, _, _ = key
        group.sort(key=lambda x: x["n_ranks"])
        baseline = group[0]

        for r in group:
            if mode == "weak":
                r["scaling_efficiency"] = (
                    baseline["time_per_step_ms"] / r["time_per_step_ms"])
            else:
                ideal = r["n_ranks"] / baseline["n_ranks"]
                actual = baseline["time_per_step_ms"] / r["time_per_step_ms"]
                r["scaling_efficiency"] = actual / ideal

    return results


def write_combined_csv(results: list[dict], path: Path) -> None:
    """Write all results to a single CSV."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  CSV: {path}")


def print_summary_table(results: list[dict]) -> None:
    """Print a summary table to stdout."""
    if not results:
        return

    print(f"\n{'='*100}")
    print(f"  CPU MPI SCALING SUMMARY")
    print(f"{'='*100}")

    header = (
        f"{'Grid':>14s} | {'Physics':>12s} | {'Mode':>6s} | "
        f"{'Prec':>7s} | {'Ranks':>5s} | {'Res':>5s} | {'L':>3s} | "
        f"{'ms/step':>9s} | {'SYPD':>8s} | {'Mcell/s':>9s} | {'Eff':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (
        x["grid_type"], x["physics_level"], x["mode"],
        x["resolution"], x["n_ranks"]
    )):
        print(
            f"{r['grid_type']:>14s} | {r['physics_level']:>12s} | "
            f"{r['mode']:>6s} | {r['precision']:>7s} | "
            f"{r['n_ranks']:>5d} | {r['resolution']:<5d} | "
            f"{r['n_levels']:>3d} | "
            f"{r['time_per_step_ms']:>9.2f} | {r['sypd']:>8.3f} | "
            f"{r['mcells_per_s']:>9.1f} | "
            f"{r.get('scaling_efficiency', 1.0):>5.1%}"
        )
    print()


def plot_results(results: list[dict], output_dir: Path) -> None:
    """Generate scaling plots per grid type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    grid_types = sorted(set(r["grid_type"] for r in results))
    physics_colors = {
        "held_suarez": "#2196F3",
        "gray_sbm": "#F44336",
        "rrtmg_full": "#4CAF50",
    }
    physics_markers = {
        "held_suarez": "o",
        "gray_sbm": "s",
        "rrtmg_full": "D",
    }

    for grid in grid_types:
        grid_results = [r for r in results if r["grid_type"] == grid]

        # --- Weak scaling plot ---
        weak = [r for r in grid_results if r["mode"] == "weak"]
        if weak:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
            for phys in sorted(set(r["physics_level"] for r in weak)):
                data = sorted(
                    [r for r in weak if r["physics_level"] == phys],
                    key=lambda r: r["n_ranks"],
                )
                ranks = [r["n_ranks"] for r in data]
                ms = [r["time_per_step_ms"] for r in data]
                ax.plot(ranks, ms,
                        color=physics_colors.get(phys, "#333"),
                        marker=physics_markers.get(phys, "o"),
                        linewidth=2, markersize=7, label=phys)

            ax.set_xscale("log", base=2)
            ax.set_xlabel("MPI Ranks", fontsize=12)
            ax.set_ylabel("Time per step [ms]", fontsize=12)
            ax.set_title(f"Weak Scaling -- {grid} (CPU MPI)", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = output_dir / f"weak_{grid}.png"
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"  Plot: {path}")

        # --- Strong scaling plot ---
        strong = [r for r in grid_results if r["mode"] == "strong"]
        if strong:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
            resolutions = sorted(set(r["resolution"] for r in strong))
            res_colors = {
                res: plt.cm.Set1(i / max(len(resolutions), 1))
                for i, res in enumerate(resolutions)
            }

            for res in resolutions:
                for phys in sorted(set(r["physics_level"] for r in strong)):
                    data = sorted(
                        [r for r in strong
                         if r["resolution"] == res and r["physics_level"] == phys],
                        key=lambda r: r["n_ranks"],
                    )
                    if not data:
                        continue
                    ranks = [r["n_ranks"] for r in data]
                    sypd = [r["sypd"] for r in data]
                    label = f"res={res} {phys}"
                    ax.plot(ranks, sypd,
                            color=res_colors[res],
                            marker=physics_markers.get(phys, "o"),
                            linewidth=2, markersize=7, label=label)

            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=10)
            ax.set_xlabel("MPI Ranks", fontsize=12)
            ax.set_ylabel("SYPD", fontsize=12)
            ax.set_title(f"Strong Scaling -- {grid} (CPU MPI)", fontsize=13)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = output_dir / f"strong_{grid}.png"
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"  Plot: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate CPU MPI scaling results.")
    parser.add_argument("results_dir", type=str,
                        help="Directory containing JSON result files.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    if not base_dir.exists():
        print(f"ERROR: {base_dir} does not exist")
        return 1

    results = load_results(base_dir)
    if not results:
        print("No results found.")
        return 0

    print(f"Loaded {len(results)} results from {base_dir}")

    results = compute_scaling_efficiency(results)
    print_summary_table(results)
    write_combined_csv(results, base_dir / "all_cpu_scaling.csv")

    if not args.no_plot:
        plot_results(results, base_dir / "plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
