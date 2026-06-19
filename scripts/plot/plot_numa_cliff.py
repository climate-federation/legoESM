"""Plot the MPAS np16->np32 cliff and its fix: SYPD vs cores, packed vs hybrid.

Reads the raw per-case JSONs (NOT the deduped tidy CSV, which collapses packed
and hybrid at matched cores) for one grid/resolution/precision and draws one
line per per-node config (cpus_per_task), so the packed 32r x 1c cliff and the
hybrid 8r x 4c recovery are both visible.

    python scripts/plot/plot_numa_cliff.py --root results/bcw_scaling \
        --grid icosahedral --resolution 5 --precision float64 \
        --out docs/scaling/mpas_numa_cliff.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(root: Path, grid: str, res: int, prec: str, mode: str):
    # cpus_per_task -> {n_cores: max sypd}
    by_cfg: dict = defaultdict(dict)
    for jf in Path(root).rglob("*.json"):
        # CPU-ladder dirs only: skip A/B, validation, and GPU-job dirs (a GPU
        # job that CUDA-fell-back to CPU lands under _gpu_ with backend=cpu and
        # a non-comparable node spread).
        s = str(jf)
        if "/_ab_" in s or "/val_" in s or "_gpu_" in s:
            continue
        try:
            d = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (d.get("grid_type") != grid or d.get("resolution") != res
                or d.get("precision") != prec or d.get("mode") != mode
                or d.get("sypd") is None):
            continue
        # The NUMA / rank-count cliff is CPU-specific; GPU runs also report
        # cpus_per_task=1 and would contaminate the packed line.
        backend = str(d.get("backend", "")).lower()
        if not backend:
            backend = "gpu" if "_gpu_" in str(jf) else "cpu"
        if backend != "cpu":
            continue
        cpt = int(d.get("cpus_per_task") or 1)
        n_ranks = d.get("n_ranks") or d.get("n_gpus") or 0
        cores = int(d.get("n_cores") or (n_ranks * cpt))
        if cores <= 0:
            continue
        prev = by_cfg[cpt].get(cores)
        if prev is None or d["sypd"] > prev:
            by_cfg[cpt][cores] = d["sypd"]
    return by_cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/bcw_scaling")
    p.add_argument("--grid", default="icosahedral")
    p.add_argument("--resolution", type=int, default=5)
    p.add_argument("--precision", default="float64")
    p.add_argument("--mode", default="strong")
    p.add_argument("--out", default="docs/scaling/mpas_numa_cliff.png")
    args = p.parse_args()

    by_cfg = collect(Path(args.root), args.grid, args.resolution,
                     args.precision, args.mode)
    if not by_cfg:
        print("no matching rows")
        return 1
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    _label = {1: "packed (1 core/rank)", 2: "hybrid 2 cores/rank",
              4: "hybrid 4 cores/rank", 8: "hybrid 8 cores/rank"}
    for cpt in sorted(by_cfg):
        pts = sorted(by_cfg[cpt].items())
        xs = [c for c, _ in pts]
        ys = [s for _, s in pts]
        ax.plot(xs, ys, "o-", lw=1.8, ms=6,
                label=_label.get(cpt, f"{cpt} cores/rank"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("CPU cores")
    ax.set_ylabel("SYPD")
    ax.set_title(
        f"MPAS {args.grid} I{args.resolution} {args.mode} {args.precision}: "
        f"np->cores cliff vs hybrid fix", fontsize=10, loc="left")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")
    for cpt in sorted(by_cfg):
        print(f"  cpt={cpt}: {sorted(by_cfg[cpt].items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
