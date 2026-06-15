"""Plot the EVOLUTION of the main scaling indicators across campaign
iterations (atm + ocean × MPI/GPU × weak/strong).

Reads the append-only tracked ledger ``docs/scaling/scaling_indicators.csv``
(date,commit,tag,grid,backend,mode,metric,value,unit,job,note) and plots
each indicator's value vs iteration order, so progress over time is
visible at a glance.  Regenerated EVERY campaign iteration (per the
2026-06-13 directive); append a row to the CSV when a new measurement
lands, then re-run this.

Usage: plot_scaling_indicators.py [out_png]
       (default results/scaling_ginsburg/scaling_indicators_evolution.png)
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEDGER = Path("docs/scaling/scaling_indicators.csv")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "results/scaling_ginsburg/scaling_indicators_evolution.png")

# Each panel groups indicators that share units/direction.  "better"
# annotates the improvement direction so the plot reads unambiguously.
PANELS = [
    ("Ocean halo exchanges / step", "exch_per_step", "lower better",
     ["ocean_latlon", "atm_latlon"]),
    ("Strong-scaling efficiency (np8)", "eff_np8", "higher better", None),
    ("Weak-scaling growth np1->8", "growth_np1to8", "lower better (1=flat)",
     None),
    ("GPU 2-device strong/weak eff", "eff_2gpu", "higher better (PCIe~0.73)",
     None),
    ("Multinode SPMD speedup (np6)", "speedup_np6", "higher better", None),
    ("Ocean barotropic precond speedup", "speedup", "higher better (1=neutral)",
     None),
    ("CPU tridiag LAPACK speedup", "tridiag", "higher better (Ginsburg CPU)",
     None),
    ("Cube PE halo collectives cut/substep", "collective_cut",
     "higher=more saved (multinode/MPI)", None),
    ("Cube SW dycore max validated devices", "tiled_np_validated",
     "higher = >6-device unlock (sub-face tiling)", None),
    ("Cube 3D-PE dycore ops np24-tiled", "ops3d_np24",
     "higher = more fv3_hydrostatic ops sub-face-tiled", None),
    ("Ocean RK3 momentum overhead (ms)", "rk3_overhead_ms",
     "lower better (per-step compute; reuse frozen EOS/pressure)", None),
    ("Ocean barotropic reductions/step", "baro_reductions_per_step",
     "lower better (jacobi120-unconverged vs banded-MG24; weak Amdahl term)",
     None),
]


def _load():
    rows = []
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            try:
                r["value"] = float(r["value"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def main() -> None:
    rows = _load()
    # iteration order = sorted unique (date, tag); x position by first
    # appearance so all panels share one timeline.
    order = []
    for r in rows:
        key = (r["date"], r["tag"])
        if key not in order:
            order.append(key)
    xpos = {k: i for i, k in enumerate(order)}
    xlabels = [f"{d[5:]}\n{t}" for (d, t) in order]

    fig, axes = plt.subplots(4, 3, figsize=(22, 16.5))
    for ax in axes.flat[len(PANELS):]:
        ax.set_visible(False)            # hide unused grid slots
    for ax, (title, metric_pref, direction, grids) in zip(
            axes.flat, PANELS):
        series = defaultdict(list)  # label -> [(x, value)]
        for r in rows:
            if not r["metric"].startswith(metric_pref):
                continue
            if grids is not None and r["grid"] not in grids:
                continue
            label = f"{r['grid']}/{r['backend']}:{r['metric']}"
            series[label].append((xpos[(r["date"], r["tag"])], r["value"]))
        for label, pts in sorted(series.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-o", ms=6, lw=1.8, label=label)
        ax.set_title(f"{title}\n({direction})", fontsize=9.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xlabels, fontsize=6.5, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        if series:
            ax.legend(fontsize=6.5)
        else:
            ax.text(0.5, 0.5, "no data yet", ha="center",
                    transform=ax.transAxes)
    fig.suptitle(
        "legoESM scaling-indicator evolution across campaign iterations "
        "(atm + ocean x MPI/GPU x weak/strong)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
