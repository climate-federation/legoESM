"""Render legoESM scaling plots.

Produces ``results/scaling/scaling.png`` with:

  * A bar chart of single-device CPU vs GPU throughput per grid type.
  * Strong-scaling curves (steps/s vs total cells) for each grid type
    when the corresponding sweep numbers are available.

The inline data dictionaries below are the canonical record so the
plot is reproducible even after the underlying npz artifacts are
pruned.  Update the dicts as new sweep cells land — or pass
``--ingest output/baroclinic_wave_diagnostics_*.npz`` to ingest a
fresh sweep automatically (iter-214 addition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Single-device steps/s for the canonical 2-day BCW (used by the bar chart).
# The CS row uses C48 — C24 hits a resolution-specific resonance documented
# in scaling.md §3.b.
SINGLE_DEVICE = [
    # (grid, resolution, ncells, dt_s, cpu_sps, gpu_sps, gpu_sps_scan24)
    # iter-212 honest re-measurement: prior iter-204/205 numbers were
    # inflated upper bounds from integer-second wall-clock timing.
    # All cells now use ``elapsed_total:.3f`` precision.
    ("spectral",     "T21",    1024,  600.0,  38.2,  198.7,  288.4),
    ("icosahedral",  "I4",     2562,  150.0,  98.8,  202.2,  205.5),
    ("cubed-sphere", "C48",   13824,  150.0,  20.2,   86.0,  103.0),
]

# Strong-scaling curves: per (grid, backend) a list of (ncells, steps_per_sec).
# Iter-212 honest re-measurement.  Note that ``--scan-steps=24`` HURTS
# icosahedral I5 (-9 %) — the dycore is already well fused and scan
# adds compile overhead with no offsetting dispatch-amortisation gain.
STRONG_SCALING = {
    "spectral": {
        "cpu":         [(8192,  38.2), (32258,   4.9)],
        "gpu":         [(8192, 198.7), (32258,  91.3)],
        "gpu_scan24":  [(8192, 288.4), (32258, 117.6)],
    },
    "icosahedral": {
        "cpu":         [(2562,  98.8), (10242,  19.1)],
        "gpu":         [(2562, 202.2), (10242, 198.5)],
        "gpu_scan24":  [(2562, 205.5), (10242, 181.1)],
    },
    "cubed-sphere": {
        "cpu":         [(13824, 20.2)],
        "gpu":         [(13824, 86.0), (55296, 65.9)],
        "gpu_scan24":  [(13824, 103.0), (55296, 67.0)],
    },
}

GRID_COLORS = {
    "spectral":     "#5B6CFF",
    "icosahedral":  "#3DCC8B",
    "cubed-sphere": "#FF8B5B",
}


def _bar_chart(ax):
    grids = [r[0] for r in SINGLE_DEVICE]
    res = [r[1] for r in SINGLE_DEVICE]
    cpu = np.array([r[4] for r in SINGLE_DEVICE])
    gpu = np.array([r[5] for r in SINGLE_DEVICE])
    gpu_scan = np.array([r[6] for r in SINGLE_DEVICE])
    speedup = gpu / cpu
    speedup_scan = np.where(gpu_scan > 0, gpu_scan / cpu, np.nan)
    x = np.arange(len(SINGLE_DEVICE))
    w = 0.27
    ax.bar(x - w, cpu, w, label="CPU (8 threads)", color="#5B6CFF")
    ax.bar(x,     gpu, w, label="GPU (default loop)", color="#FF8B5B")
    ax.bar(x + w, gpu_scan, w, label="GPU (--scan-steps=24)", color="#FF4500")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{r}" for g, r in zip(grids, res)])
    ax.set_ylabel("BCW steps / second")
    ax.set_title("Single-device throughput (warm step)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (c_, g_, gs_, s_, ss_) in enumerate(
        zip(cpu, gpu, gpu_scan, speedup, speedup_scan)
    ):
        ax.text(i - w, c_ + 5, f"{c_:.0f}", ha="center", fontsize=8)
        ax.text(i,     g_ + 5, f"{g_:.0f}\n({s_:.1f}×)",
                ha="center", fontsize=8, color="#B33B00")
        if gs_ > 0:
            ax.text(i + w, gs_ + 5, f"{gs_:.0f}\n({ss_:.1f}×)",
                    ha="center", fontsize=8, color="#B33B00", fontweight="bold")


def _strong_curves(ax):
    for grid, by_backend in STRONG_SCALING.items():
        color = GRID_COLORS[grid]
        for backend, marker, dash, lw in [
            ("cpu",        "o", "--", 1.2),
            ("gpu",        "s", "-",  1.5),
            ("gpu_scan24", "D", "-",  2.2),
        ]:
            pts = sorted(by_backend.get(backend, []))
            if not pts:
                continue
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            label = f"{grid} {backend.upper()}"
            ax.plot(xs, ys, marker + dash, color=color,
                    label=label, linewidth=lw, markersize=7,
                    alpha=0.95 if backend == "gpu_scan24" else 0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total cells")
    ax.set_ylabel("BCW steps / second")
    ax.set_title("Strong scaling (single device)")
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)


def _ingest_npz_glob(pattern: str) -> dict:
    """Walk NPZ files matching ``pattern`` and overwrite STRONG_SCALING /
    SINGLE_DEVICE entries when a matching (grid, resolution, backend)
    cell is found.  Tags are interpreted by the ``parse_strong_sweep``
    convention: ``..._<backend>_<grid>_<res>``.  Throughput is read from
    the iter-202 ``steps_per_sec`` field (npz-stored).
    """
    import glob
    import re
    overrides = []
    for p in glob.glob(pattern):
        m = re.search(r"_(cpu|gpu)_(spectral|icosahedral|cubed-sphere)_(\d+)\.npz$", p)
        if not m:
            continue
        backend, grid, res = m.group(1), m.group(2), int(m.group(3))
        d = np.load(p, allow_pickle=True)
        if "steps_per_sec" not in d.files:
            continue
        sps = float(d["steps_per_sec"])
        ncells = _grid_ncells(grid, res)
        overrides.append((grid, backend, ncells, sps))
    return overrides


def _grid_ncells(grid: str, res: int) -> int:
    if grid == "spectral":
        n_lat = 3 * res + 1
        return 2 * n_lat * n_lat
    if grid == "icosahedral":
        return 10 * 4 ** res + 2
    if grid == "cubed-sphere":
        return 6 * res * res
    raise ValueError(grid)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest", default=None,
        help="Glob over NPZ files (e.g. 'output/baroclinic_wave_diagnostics_iter214_*.npz') "
             "to override the inline STRONG_SCALING gpu/gpu_scan24 entries.",
    )
    args = parser.parse_args()

    if args.ingest:
        for grid, backend, ncells, sps in _ingest_npz_glob(args.ingest):
            key = "gpu" if backend == "gpu" else "cpu"
            curve = STRONG_SCALING.get(grid, {}).get(key, [])
            # Replace existing entry at this ncells, otherwise append.
            curve = [(c, s) for c, s in curve if c != ncells]
            curve.append((ncells, sps))
            STRONG_SCALING.setdefault(grid, {})[key] = curve
            print(f"  ingested {grid}/{backend}/n={ncells} → {sps:.1f} sps")

    out_dir = Path("results/scaling")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    _bar_chart(axes[0])
    _strong_curves(axes[1])
    fig.suptitle("legoESM scaling — single-device CPU vs GPU "
                 "(latest, 2026-05-03)", fontsize=12, fontweight="bold")
    out_path = out_dir / "scaling.png"
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
