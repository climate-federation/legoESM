#!/usr/bin/env python
"""Overlay legoESM ocean scaling on Oceananigans reference curves.

Reference: Silvestri et al. (CliMA Oceananigans, Julia), quasi-realistic
global ocean ("Double Drake"-style), DOUBLE precision, 100 vertical
levels, NVIDIA A100s on Perlmutter. Figure 10 = strong scaling (wall
time per step + SYPD vs GPUs, per-resolution ladders 1/4..1/96 deg,
each with its own dt). Figure 11 = weak scaling (flat ~0.85 s/step from
4 -> 1024 GPUs at a fixed ~1/12 deg-equivalent patch per GPU).

Anchors below were digitized BY EYE from screenshots — +/-30%, not data.

Comparison basis: SYPD embeds each model's dt choice, so the figure also
plots throughput (Mcells/s incl. vertical) which is dt-free:
  Oceananigans 1/4 deg @ 1 A100: 1440*720*100 cells / 0.17 s ~ 610 Mcells/s.
Hardware caveat (printed on figure): A100 vs Quadro RTX 8000 — ~19x f64
FLOPs (Turing runs f64 at 1/32 rate), ~2.6x HBM bandwidth.

legoESM inputs: ocean_scaling.csv files from
``scripts/bench/bench_ocean_gpu_scaling.py`` (TimingResult schema).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Strong scaling anchors digitized from Fig. 10 (A100, f64, 100 levels):
# resolution_deg: (total_cells_incl_vertical, dt_s, [(n_gpus, ms_per_step)])
OCEANANIGANS_STRONG = {
    "1/4": (1440 * 720 * 100, 900.0, [(1, 170.0), (8, 25.0)]),
    "1/8": (2880 * 1440 * 100, 420.0, [(4, 400.0), (32, 50.0)]),
    "1/12": (4320 * 2160 * 100, 270.0, [(8, 400.0), (64, 60.0)]),
    "1/48": (17280 * 8640 * 100, 67.5, [(64, 800.0), (512, 100.0)]),
    "1/96": (34560 * 17280 * 100, 33.75, [(256, 800.0), (1024, 300.0)]),
}
# Weak scaling headline from Fig. 11: ~0.85 s/step, flat 4 -> 1024 GPUs at
# fixed ~1/12-deg-equivalent per-GPU patch (used as a reference line only).
OCEANANIGANS_WEAK_MS = 850.0

SECONDS_PER_YEAR = 365.0 * 86400.0


def sypd(ms_per_step: float, dt_s: float) -> float:
    steps_per_year = SECONDS_PER_YEAR / dt_s
    return 86400.0 / (steps_per_year * ms_per_step / 1e3)


def load_rows(result_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in result_dirs:
        for csv_path in sorted(root.rglob("ocean_scaling.csv")):
            with csv_path.open() as fh:
                for row in csv.DictReader(fh):
                    row["_source"] = str(csv_path)
                    rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--output", type=Path,
                    default=Path("results/scaling_ginsburg/ocean_vs_oceananigans.png"))
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"no ocean_scaling.csv rows under {args.results}")

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 5.5))

    cmap = plt.get_cmap("tab10")
    for i, (label, (cells, dt_s, pts)) in enumerate(OCEANANIGANS_STRONG.items()):
        xs = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        ax_t.plot(xs, ms, "--o", color=cmap(i), alpha=0.7,
                  label=f"Oceananigans {label}° (A100 f64)")
        ax_s.plot(xs, [sypd(m, dt_s) for m in ms], "--o", color=cmap(i),
                  alpha=0.7, label=f"Oceananigans {label}°")
        # per-device throughput annotation at first anchor
        tput = cells / (ms[0] / 1e3) / xs[0] / 1e6
        ax_t.annotate(f"{tput:.0f} Mc/s/GPU", (xs[0], ms[0]), fontsize=6,
                      textcoords="offset points", xytext=(3, 4), color=cmap(i))

    # legoESM rows: group by (grid id prefix, precision); x = n_gpus.
    groups: dict[tuple, list[tuple[int, float, float, float]]] = defaultdict(list)
    for r in rows:
        try:
            n_dev = int(r.get("n_gpus", 1))
            ms_step = float(r["time_per_step_ms"])
            res = r.get("resolution", "?")
            cells = float(r.get("total_cells", 0))
            sy = float(r.get("sypd", 0))
        except (KeyError, ValueError):
            continue
        groups[(res, r.get("precision", "?"))].append((n_dev, ms_step, cells, sy))

    for (res, prec), pts in sorted(groups.items()):
        pts = sorted(set(pts))
        xs = [p[0] for p in pts]
        ms = [p[1] for p in pts]
        sys_ = [p[3] for p in pts]
        ls = "-" if prec == "float32" else ":"
        line, = ax_t.plot(xs, ms, ls, marker="s",
                          label=f"legoESM {res} {prec} (RTX8000)")
        ax_s.plot(xs, sys_, ls, marker="s", color=line.get_color(),
                  label=f"legoESM {res} {prec}")
        if pts[0][2] > 0:
            tput = pts[0][2] / (ms[0] / 1e3) / xs[0] / 1e6
            ax_t.annotate(f"{tput:.0f} Mc/s/GPU", (xs[0], ms[0]), fontsize=6,
                          textcoords="offset points", xytext=(3, -8),
                          color=line.get_color())

    ax_t.axhline(OCEANANIGANS_WEAK_MS, color="0.5", lw=0.8, ls=":",
                 label="Oceananigans weak-scaling plateau")
    for ax, ylab in ((ax_t, "wall time per step [ms]"), (ax_s, "SYPD")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("legoESM ocean vs Oceananigans (Silvestri et al., Fig. 10/11)")
    fig.text(0.01, 0.01,
             "Oceananigans anchors digitized by eye (+/-30%); A100 vs RTX 8000: "
             "~19x f64 FLOPs, ~2.6x BW per device. Oceananigans: 100 vertical "
             "levels; legoESM bench: 20. SYPD embeds each model's dt — "
             "throughput annotations (Mcells/s incl. vertical) are dt-free.",
             fontsize=6, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
