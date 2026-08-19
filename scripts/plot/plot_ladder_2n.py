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

# (label prefix, receipt-filename glob, colour, marker). The problem size in
# each label is READ FROM THE RECEIPT, never typed here -- a hand-typed cell
# count is one refactor away from labelling the wrong mesh.
# Several filename conventions exist because the job scripts were renamed
# mid-campaign; a point re-measured under the newer name must not be missed,
# so each series lists every glob that has ever held its receipts and the
# HIGHEST job id wins for a given device count.
SERIES = [
    ("MPAS L9", ("mpas_n*.jsonl", "mpas_s9_n*.jsonl"), "#0072B2", "o"),
    # Same mesh, same exchange, only the partitioner differs — the one
    # confirmed win of the campaign, measured 5.5 to 19.3% in controlled
    # A/Bs and re-run here in the ladder's own configuration so the two
    # curves are comparable.
    ("MPAS L9, graph partition", ("mpas_s9_metis_n*.jsonl",),
     "#009E73", "v"),
    ("MPAS L10", ("mpas_s10_n*.jsonl",), "#56B4E9", "s"),
    ("lat-lon", ("latlon_n*.jsonl",), "#D55E00", "^"),
    ("lat-lon", ("latlon_4096_n*.jsonl",), "#E69F00", "D"),
]

# Levante's gpu partition holds 63 A100 nodes x 4 GPUs = 252 devices, so 128
# is the largest power of two that fits. 256 and 512 are not reachable here.
MACHINE_MAX_POW2 = 128


def _size_label(rec):
    """Problem size straight from the receipt, never typed by hand."""
    nlev = rec.get("nlev") or rec.get("n_levels")
    if rec.get("n_cells") and nlev:
        return f"{rec['n_cells']:,} cells x {nlev} lev"
    if rec.get("n_lat") and rec.get("n_lon") and nlev:
        return f"{rec['n_lat']}x{rec['n_lon']} x {nlev} lev"
    return None


def load(receipt_dir: str, patterns):
    """Return ([(n_devices, ms, job_id)] sorted by device count, size label).

    A file may hold several lines when a point was re-measured; the LAST line
    is that file's current receipt.  When two FILES carry the same device
    count (a re-measurement landed under a newer filename), the higher SLURM
    job id wins -- job ids increase with time on this cluster.
    """
    best = {}
    size = None
    for pattern in patterns:
        for path in glob.glob(os.path.join(receipt_dir, pattern)):
            with open(path) as fh:
                lines = [ln for ln in fh if ln.strip()]
            if not lines:
                continue
            rec = json.loads(lines[-1])
            ms = rec.get("steady_median_ms")
            n = rec.get("n_devices")
            if ms is None or n is None:
                raise ValueError(
                    f"{path}: receipt has no steady_median_ms/n_devices")
            job = rec.get("metadata", {}).get("slurm_job_id")
            key = int(job) if job and str(job).isdigit() else -1
            if n not in best or key > best[n][0]:
                best[n] = (key, float(ms), job)
            if size is None:
                size = _size_label(rec)
    points = sorted((n, v[1], v[2]) for n, v in best.items())
    return points, size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", default="/scratch/b/b381103/legoesm_scaling/ladder2n")
    ap.add_argument("--out", default="ladder2n.png")
    args = ap.parse_args()

    fig, (ax, axe) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for label, pattern, color, marker in SERIES:
        pts, size = load(args.receipts, pattern)
        if not pts:
            print(f"[skip] no receipts for {pattern}")
            continue
        if size:
            label = f"{label} ({size})"
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
