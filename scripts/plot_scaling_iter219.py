#!/usr/bin/env python
"""Render iter-219 scaling plots with scan-steps amortisation effect.

Reads ``results/scaling/iter219_throughput.csv`` and produces:

  results/scaling/scaling.png

A 4-panel figure showing the iter-219 step-batching speed-up:
  (a) GPU throughput at scan-steps=1 vs scan-steps=24 per grid
  (b) Speed-up factor of scan-steps=24 vs scan-steps=1 (kernel-launch
      amortisation gain)
  (c) GPU/CPU speedup factor (with --scan-steps=24)
  (d) iter-217 ref vs iter-219 measured (regression-vs-improvement check)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# iter-217 reference numbers from scaling.md TL;DR
ITER217_REF = {
    ("spectral",      "T21"): {"cpu": 38.2,  "gpu": 288.4},
    ("icosahedral",   "I4"):  {"cpu": 98.8,  "gpu": 205.5},
    ("cubed-sphere",  "C24"): {"cpu": 20.2,  "gpu":  None},
}


def _normalise_res(grid: str, res: str) -> str:
    res = res.upper()
    if grid == "icosahedral" and res.isdigit():
        return f"I{res}"
    if grid == "cubed-sphere" and res.isdigit():
        return f"C{res}"
    return res


def load(csv_path: Path):
    data: dict[tuple[str, str], dict[tuple[str, int], float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            grid = row["grid"]
            res = _normalise_res(grid, row["resolution"])
            backend = row["backend"]
            try:
                ss = int(row.get("scan_steps", 1) or 1)
            except (ValueError, TypeError):
                ss = 1
            try:
                sps = float(row["steps_per_sec"]) if row["steps_per_sec"] else None
            except ValueError:
                sps = None
            data.setdefault((grid, res), {})[(backend, ss)] = sps
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path,
                        default=Path("results/scaling/iter219_throughput.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/scaling/scaling.png"))
    args = parser.parse_args(argv)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    data = load(args.csv)
    keys = sorted(data.keys())

    grid_colors = {
        "spectral":     "#1f77b4",
        "icosahedral":  "#2ca02c",
        "cubed-sphere": "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "legoESM iter-219 BCW scaling — `--scan-steps` kernel-launch amortisation",
        fontsize=13,
    )
    ax_thr, ax_ss, ax_sp, ax_ref = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    labels = [f"{g}/{r}" for g, r in keys]
    bar_x = np.arange(len(keys))
    width = 0.35

    # (a) GPU ss=1 vs ss=24 throughput
    gpu1 = [data[k].get(("gpu", 1)) or np.nan for k in keys]
    gpu24 = [data[k].get(("gpu", 24)) or np.nan for k in keys]
    ax_thr.bar(bar_x - width / 2, gpu1, width, label="--scan-steps=1",
                color="#cccccc")
    ax_thr.bar(bar_x + width / 2, gpu24, width, label="--scan-steps=24",
                color=[grid_colors.get(g, "#000000") for g, _ in keys])
    ax_thr.set_ylabel("GPU throughput [steps/s]")
    ax_thr.set_xticks(bar_x)
    ax_thr.set_xticklabels(labels, rotation=20, ha="right")
    ax_thr.set_title("(a) GPU throughput vs scan-batch size")
    ax_thr.legend()
    ax_thr.grid(True, axis="y", alpha=0.3)
    for i, (b, a) in enumerate(zip(gpu1, gpu24)):
        if not np.isnan(b) and not np.isnan(a):
            ax_thr.text(i, max(b, a) * 1.04, f"{a:.0f} sps",
                          ha="center", fontsize=8)

    # (b) Speed-up of ss=24 vs ss=1
    ratios = []
    for k in keys:
        a = data[k].get(("gpu", 24))
        b = data[k].get(("gpu", 1))
        ratios.append((a / b) if (a and b) else np.nan)
    ax_ss.bar(bar_x, ratios, color=[grid_colors.get(g, "#000000")
                                    for g, _ in keys])
    ax_ss.axhline(1.0, color="gray", lw=1, ls="--")
    ax_ss.set_ylabel("scan-steps=24 / scan-steps=1")
    ax_ss.set_xticks(bar_x)
    ax_ss.set_xticklabels(labels, rotation=20, ha="right")
    ax_ss.set_title("(b) Kernel-launch amortisation gain (this iter)")
    for i, r in enumerate(ratios):
        if not np.isnan(r):
            ax_ss.text(i, r * 1.02, f"{r:.2f}×", ha="center", fontsize=10)
    ax_ss.grid(True, axis="y", alpha=0.3)

    # (c) GPU/CPU speed-up — uses scan-steps=24 GPU and scan-steps=1 CPU
    sp = []
    for k in keys:
        g = data[k].get(("gpu", 24)) or data[k].get(("gpu", 1))
        c = data[k].get(("cpu", 1))
        sp.append((g / c) if (g and c) else np.nan)
    ax_sp.bar(bar_x, sp, color=[grid_colors.get(g, "#000000")
                                  for g, _ in keys])
    ax_sp.axhline(1.0, color="gray", lw=1, ls="--")
    ax_sp.set_ylabel("GPU(ss=24) / CPU(ss=1)")
    ax_sp.set_xticks(bar_x)
    ax_sp.set_xticklabels(labels, rotation=20, ha="right")
    ax_sp.set_title("(c) Best-GPU vs CPU speed-up")
    for i, s in enumerate(sp):
        if not np.isnan(s):
            ax_sp.text(i, s * 1.02, f"{s:.1f}×", ha="center", fontsize=10)
    ax_sp.grid(True, axis="y", alpha=0.3)

    # (d) iter-217 ref vs iter-219 measured
    ref_x = []
    ref_gpu, cur_gpu, cmp_lab = [], [], []
    for k in keys:
        ref = ITER217_REF.get(k)
        if ref is None or ref.get("gpu") is None:
            continue
        ref_gpu.append(ref["gpu"])
        cur_gpu.append(data[k].get(("gpu", 24)) or data[k].get(("gpu", 1)))
        cmp_lab.append(f"{k[0]}/{k[1]}")
    if cmp_lab:
        x = np.arange(len(cmp_lab))
        ax_ref.bar(x - width / 2, ref_gpu, width, label="iter-217 ref",
                    color="#999999")
        ax_ref.bar(x + width / 2, cur_gpu, width, label="iter-219 (this)",
                    color="#1f77b4")
        ax_ref.set_ylabel("GPU throughput [steps/s]")
        ax_ref.set_xticks(x)
        ax_ref.set_xticklabels(cmp_lab, rotation=20, ha="right")
        ax_ref.set_title("(d) iter-217 ref vs iter-219 measurement")
        ax_ref.legend()
        ax_ref.grid(True, axis="y", alpha=0.3)
        for i, (r, c) in enumerate(zip(ref_gpu, cur_gpu)):
            if r and c:
                pct = (c - r) / r * 100
                ax_ref.text(i, max(r, c) * 1.02,
                              f"{pct:+.0f}%",
                              ha="center", va="bottom", fontsize=10,
                              color="green" if pct >= 0 else "red")
    else:
        ax_ref.text(0.5, 0.5, "no overlap with iter-217 ref",
                     ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
