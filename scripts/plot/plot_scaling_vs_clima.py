#!/usr/bin/env python
"""Overlay legoESM scaling results on CliMA JAMES Fig. 13 reference curves.

CliMA reference: SYPD vs device count for the *moist baroclinic wave*
benchmark (no full physics) at 103/51/26/13 km horizontal resolution,
GPU panel = NVIDIA H100 (solid) / A100 (dashed), CPU panel = Caltech
Resnick cluster. Reference points below were digitized by eye from a
screenshot of Figure 13 — treat as +/-30% anchors, not data.

legoESM inputs: TimingResult CSVs written by
``scripts/bench/run_levante_gpu_scaling.py`` (``all_scaling.csv`` /
``strong_scaling.csv`` / ``weak_scaling.csv``) under one or more result
directories. Cubed-sphere resolution -> approx grid spacing at the
equator: 90/N degrees per cell -> ~10008/N km (C48~208 km, C96~104 km,
C192~52 km), i.e. C96/C192 line up with the CliMA 103/51 km curves.

Honest-comparison caveats (printed onto the figure):
- Hardware: Quadro RTX 8000 (Turing) vs H100/A100 — roughly 5-7x peak
  float32 FLOPs and ~5x HBM bandwidth in CliMA's favor per device.
- Physics tier: legoESM curves come in two bounds — dry JW baroclinic
  wave (physics=none, cheaper than CliMA's moist wave) and gray_sbm
  AMIP pipeline (radiation+convection+microphysics, heavier).
- Vertical levels: legoESM runs use 26 levels; CliMA's level count in
  Fig. 13 is not stated on the figure. SYPD scales ~1/n_levels.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Digitized-by-eye anchors from CliMA JAMES Fig. 13 (see module docstring).
# {resolution_km: [(n_devices, sypd), ...]}
CLIMA_GPU_H100 = {
    103: [(1, 6.0), (8, 35.0), (64, 100.0)],
    51: [(1, 1.5), (8, 9.0), (64, 30.0), (512, 60.0)],
    26: [(1, 0.35), (16, 4.0), (128, 12.0)],
    13: [(8, 0.5), (512, 2.5)],
}
CLIMA_CPU = {
    103: [(16, 4.0), (256, 50.0)],
    51: [(16, 0.6), (256, 7.0)],
    26: [(16, 0.12), (256, 1.5)],
    13: [(16, 0.012), (256, 0.15)],
}

# Approx equatorial grid spacing for cubed-sphere CN (km).
CS_KM = lambda n: 10008.0 / float(n)  # noqa: E731

RES_COLORS = {103: "tab:cyan", 51: "tab:orange", 26: "tab:red", 13: "tab:blue"}


def load_rows(result_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in result_dirs:
        for csv_path in sorted(root.rglob("*_scaling.csv")):
            if csv_path.name == "all_scaling.csv":
                continue  # strong/weak files already cover these rows
            with csv_path.open() as fh:
                for row in csv.DictReader(fh):
                    row["_source"] = str(csv_path)
                    rows.append(row)
    return rows


def nearest_clima_res(km: float) -> int | None:
    best, bestrel = None, math.inf
    for ref in RES_COLORS:
        rel = abs(km - ref) / ref
        if rel < bestrel:
            best, bestrel = ref, rel
    return best if bestrel <= 0.15 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", type=Path,
                    help="legoESM scaling result dirs (searched recursively)")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu",
                    help="Which CliMA panel to compare against")
    ap.add_argument("--mode", choices=["strong", "weak", "both"],
                    default="strong")
    ap.add_argument("--output", type=Path,
                    default=Path("results/scaling_ginsburg/vs_clima.png"))
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"no *_scaling.csv rows found under {args.results}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ref = CLIMA_GPU_H100 if args.backend == "gpu" else CLIMA_CPU
    for km, pts in sorted(ref.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "--", color=RES_COLORS[km], alpha=0.7,
                label=f"CliMA {km} km ({'H100' if args.backend == 'gpu' else 'CPU'})")

    # legoESM rows: group by (physics, precision, resolution) -> sypd vs devices.
    groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if args.mode != "both" and r.get("mode") != args.mode:
            continue
        try:
            n_dev = int(r["n_gpus"])
            sypd = float(r["sypd"])
            res = int(r["resolution"])
        except (KeyError, ValueError):
            continue
        key = (r.get("physics_level", "?"), r.get("precision", "?"), res)
        groups[key].append((n_dev, sypd))

    markers = {"none": "o", "held_suarez": "s", "gray_sbm": "^",
               "rrtmg_full": "v"}
    for (phys, prec, res), pts in sorted(groups.items()):
        pts = sorted(set(pts))
        km = CS_KM(res)
        ckm = nearest_clima_res(km)
        color = RES_COLORS.get(ckm, "0.4")
        xs, ys = zip(*pts)
        ls = "-" if prec == "float32" else ":"
        ax.plot(xs, ys, ls, marker=markers.get(phys, "x"), color=color,
                label=f"legoESM C{res} (~{km:.0f} km) {phys} {prec}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"Number of {'GPUs' if args.backend == 'gpu' else 'CPU ranks'}")
    ax.set_ylabel("SYPD")
    ax.set_title("legoESM vs CliMA moist baroclinic wave (JAMES Fig. 13)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.text(0.01, 0.01,
             "CliMA anchors digitized by eye (+/-30%). RTX 8000 vs "
             "H100: ~5-7x FLOPs, ~5x BW per device in CliMA's favor. "
             "legoESM: 26 levels; dry JW (none) < CliMA moist wave < "
             "gray_sbm AMIP in per-step cost.",
             fontsize=6, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
