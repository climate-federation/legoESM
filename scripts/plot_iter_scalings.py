#!/usr/bin/env python
"""Aggregate iter-N scaling runs into figures under ``results/scalings/``.

Reads CSV/JSON produced by ``run_levante_gpu_scaling.py`` and the per-case
JSONs produced by ``run_cpu_mpi_scaling.py``, then renders four PNGs:

* ``iterN_gpu_throughput.png`` — Mcells/s vs problem size, per grid, GPU.
* ``iterN_cpu_throughput.png`` — Mcells/s vs problem size, per grid, CPU.
* ``iterN_sypd.png`` — Simulated-years-per-day vs problem size, GPU + CPU.
* ``iterN_summary.png`` — combined dashboard (GPU vs CPU, all grids).

Usage::

    python scripts/plot_iter_scalings.py --iter 1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_gpu_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "grid": path.parent.name,
                "n_gpus": int(row["n_gpus"]),
                "resolution": row["resolution"],
                "n_levels": int(row["n_levels"]),
                "precision": row["precision"],
                "mode": row["mode"],
                "ms_per_step": float(row["time_per_step_ms"]),
                "sypd": float(row["sypd"]),
                "total_cells": int(row["total_cells"]),
                "cells_per_gpu": int(row["cells_per_gpu"]),
                "mcells_per_s": float(row["mcells_per_s"]),
                "device": "gpu",
            })
    return rows


def _read_cpu_jsons(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Accept either a list of timing rows or a single dict.
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                rows.append({
                    "grid": item.get("grid", path.parent.name.split("_")[0]),
                    "n_gpus": int(item.get("n_ranks", 1)),
                    "resolution": item.get("resolution", "?"),
                    "n_levels": int(item.get("n_levels", 0)),
                    "precision": item.get("precision", "?"),
                    "mode": item.get("mode", "single"),
                    "ms_per_step": float(item["time_per_step_ms"]),
                    "sypd": float(item["sypd"]),
                    "total_cells": int(item.get("total_cells", 0)),
                    "cells_per_gpu": int(item.get("cells_per_rank", item.get("total_cells", 0))),
                    "mcells_per_s": float(item.get("mcells_per_s", 0.0)),
                    "device": "cpu",
                })
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _series_per_grid(rows, x_key, y_key):
    """Group rows by grid and return ordered lists of (x, y, label)."""
    by_grid: dict[str, list[tuple]] = {}
    for r in rows:
        g = r.get("grid", "?")
        by_grid.setdefault(g, []).append((float(r[x_key]), float(r[y_key])))
    out = []
    for g, pts in by_grid.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((xs, ys, g))
    return out


def _plot_throughput(rows, *, title, ylabel, x_key, y_key, path, xlog=True, ylog=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = _series_per_grid(rows, x_key, y_key)
    for xs, ys, label in series:
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("cells per device")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if series:
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_summary(gpu_rows, cpu_rows, path, *, iter_label):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rows, label in zip(
        axes, (gpu_rows, cpu_rows), ("GPU (single device)", "CPU (single rank)"),
    ):
        for xs, ys, glabel in _series_per_grid(rows, "cells_per_gpu", "mcells_per_s"):
            ax.plot(xs, ys, marker="o", label=glabel)
        ax.set_xlabel("cells per device")
        ax.set_ylabel("Mcells/s")
        ax.set_title(label)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if rows:
            ax.legend(fontsize=9)
    fig.suptitle(f"legoESM iter {iter_label} scaling — throughput vs problem size")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iter", default="1", help="iteration label used in file names")
    p.add_argument("--scalings-dir", default="results/scalings",
                   help="root directory containing iter sub-folders")
    args = p.parse_args()

    scalings = Path(args.scalings_dir)
    scalings.mkdir(parents=True, exist_ok=True)

    iter_tag = f"iter{args.iter}"
    print(f"plotting {iter_tag} scalings under {scalings}")

    # GPU rows: from any sub-folder named like ``iter{N}_gpu*`` containing a
    # strong_scaling.csv (the levante driver's canonical output).
    gpu_rows: list[dict] = []
    for d in sorted(scalings.glob(f"{iter_tag}_gpu*")):
        for csv_name in ("strong_scaling.csv", "weak_scaling.csv", "all_scaling.csv"):
            rows = _read_gpu_csv(d / csv_name)
            for r in rows:
                # Derive a friendlier grid label from directory suffix.
                suffix = d.name.split("_", 1)[1] if "_" in d.name else d.name
                # iter1_gpu → cubed-sphere; iter1_gpu_ico → icosahedral, etc.
                if suffix == "gpu":
                    r["grid"] = "cubed-sphere"
                elif suffix == "gpu_ico":
                    r["grid"] = "icosahedral"
                elif suffix == "gpu_sp":
                    r["grid"] = "spectral"
                else:
                    r["grid"] = suffix.replace("gpu_", "")
            gpu_rows.extend(rows)
            if rows:
                break  # one CSV per dir

    # CPU rows: from any sub-folder named like ``iter{N}_cpu*`` (each contains
    # per-case JSON outputs).
    cpu_rows: list[dict] = []
    for d in sorted(scalings.glob(f"{iter_tag}_cpu*")):
        cpu_rows.extend(_read_cpu_jsons(d))

    print(f"  collected {len(gpu_rows)} GPU rows, {len(cpu_rows)} CPU rows")

    if gpu_rows:
        _plot_throughput(
            gpu_rows,
            title=f"iter {args.iter} — GPU single-device throughput",
            ylabel="Mcells/s",
            x_key="cells_per_gpu", y_key="mcells_per_s",
            path=scalings / f"{iter_tag}_gpu_throughput.png",
        )
        _plot_throughput(
            gpu_rows,
            title=f"iter {args.iter} — GPU single-device SYPD",
            ylabel="SYPD",
            x_key="cells_per_gpu", y_key="sypd",
            path=scalings / f"{iter_tag}_gpu_sypd.png",
            ylog=True,
        )
    if cpu_rows:
        _plot_throughput(
            cpu_rows,
            title=f"iter {args.iter} — CPU single-rank throughput",
            ylabel="Mcells/s",
            x_key="cells_per_gpu", y_key="mcells_per_s",
            path=scalings / f"{iter_tag}_cpu_throughput.png",
        )
        _plot_throughput(
            cpu_rows,
            title=f"iter {args.iter} — CPU single-rank SYPD",
            ylabel="SYPD",
            x_key="cells_per_gpu", y_key="sypd",
            path=scalings / f"{iter_tag}_cpu_sypd.png",
            ylog=True,
        )
    if gpu_rows or cpu_rows:
        _plot_summary(
            gpu_rows, cpu_rows,
            path=scalings / f"{iter_tag}_summary.png",
            iter_label=args.iter,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
