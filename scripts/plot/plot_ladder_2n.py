"""Power-of-two strong-scaling ladder for the atmosphere GPU lanes.

Reads the ladder receipts written by
``scripts/cluster/scaling_levante/ladder_2n_atm.sbatch`` (one JSONL line per
timed arm) and plots measured ms/step against device count on log-log axes
with a dashed ideal line anchored at each series' own first measured point,
plus a parallel-efficiency panel.

No hardcoded numbers: every point is read from a receipt file, and the SLURM
job id of each point is printed to stdout so any marker can be traced back.

Usage
-----
    python scripts/plot/plot_ladder_2n.py \
        --receipts /scratch/b/b381103/legoesm_scaling/ladder2n \
        --out ladder2n.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, receipt-filename glob). One curve per entry.
SERIES = [
    ("MPAS L9 (40,962 cells x 26 lev)", "mpas_n*.jsonl", "#0072B2", "o"),
    ("MPAS L10 (163,842 cells x 26 lev)", "mpas_s10_n*.jsonl", "#56B4E9", "s"),
    ("lat-lon 2048x4096 L26", "latlon_n*.jsonl", "#D55E00", "^"),
    ("lat-lon 4096x8192 L26", "latlon_4096_n*.jsonl", "#E69F00", "D"),
]

# Levante's gpu partition holds 63 A100 nodes x 4 GPUs = 252 devices, so 128
# is the largest power of two that fits. 256 and 512 are not reachable here.
MACHINE_MAX_POW2 = 128


def load(receipt_dir: str, pattern: str):
    """Return [(n_devices, ms, job_id)] sorted by device count."""
    points = []
    for path in glob.glob(os.path.join(receipt_dir, pattern)):
        with open(path) as fh:
            lines = [ln for ln in fh if ln.strip()]
        if not lines:
            continue
        rec = json.loads(lines[-1])  # last line = final receipt for that arm
        ms = rec.get("steady_median_ms")
        n = rec.get("n_devices")
        if ms is None or n is None:
            raise ValueError(f"{path}: receipt has no steady_median_ms/n_devices")
        points.append((n, float(ms), rec.get("metadata", {}).get("slurm_job_id")))
    return sorted(points)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", default="/scratch/b/b381103/legoesm_scaling/ladder2n")
    ap.add_argument("--out", default="ladder2n.png")
    args = ap.parse_args()

    fig, (ax, axe) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for label, pattern, color, marker in SERIES:
        pts = load(args.receipts, pattern)
        if not pts:
            print(f"[skip] no receipts for {pattern}")
            continue
        n = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        print(f"{label}")
        for dev, t, job in pts:
            print(f"    {dev:>4} GPU  {t:8.2f} ms   job {job}")

        ax.plot(n, ms, marker=marker, color=color, label=label, lw=1.8, ms=6)
        n0, t0 = n[0], ms[0]
        ideal = [t0 * n0 / k for k in n]
        ax.plot(n, ideal, ls="--", color=color, lw=1.0, alpha=0.55)

        eff = [100.0 * (t0 * n0 / k) / t for k, t in zip(n, ms)]
        axe.plot(n, eff, marker=marker, color=color, label=label, lw=1.8, ms=6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("ms / step")
    ax.set_title("Strong scaling, powers of two (dashed = ideal)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    axe.set_xscale("log", base=2)
    axe.axhline(100.0, color="0.4", ls=":", lw=1.0)
    axe.set_xlabel("GPUs")
    axe.set_ylabel("parallel efficiency vs first point (%)")
    axe.set_title("Parallel efficiency")
    axe.set_ylim(0, 130)
    axe.grid(True, which="both", alpha=0.25)

    for a in (ax, axe):
        a.axvline(MACHINE_MAX_POW2, color="0.3", ls="-.", lw=1.0, alpha=0.7)
        a.annotate("Levante ceiling\n(252 GPUs total)", xy=(MACHINE_MAX_POW2, 0.02),
                   xycoords=("data", "axes fraction"), fontsize=7,
                   ha="right", va="bottom", color="0.3")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
