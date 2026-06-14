#!/usr/bin/env python
"""Current-state scaling dashboard for the Ginsburg campaign (2x2 PNG).

A: atm GPU strong (cubed-sphere dry JW, true 2-GPU post sharding fix)
B: atm GPU weak (same run)
C: atm CPU-MPI icosahedral strong+weak (held_suarez, rank sweep)
D: ocean — GPU throughput vs size (the LL192 cliff) + CPU-MPI points

Inputs are the campaign's heterogeneous outputs; pass run dirs explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drybaro", default="results/scaling_ginsburg/atm_drybaro_8454921")
    ap.add_argument("--drybaro-old", default="results/scaling_ginsburg/atm_drybaro_8454397",
                    help="pre-fix run: 1-GPU rows only (used for f64 until rerun finishes)")
    ap.add_argument("--cpumpi", default="results/scaling_ginsburg/atm_cpumpi_8454267")
    ap.add_argument("--ocean-gpu", default="results/scaling_ginsburg/ocean_gpu_8454268")
    ap.add_argument("--ocean-mpi-csv", default="results/scaling_ginsburg/p2_smoke_8456021/ocean_scaling.csv")
    ap.add_argument("--output", default="results/scaling_ginsburg/dashboard.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # --- A: atm GPU strong ------------------------------------------------
    for src, tag, alpha in ((args.drybaro, "post-fix", 1.0),
                            (args.drybaro_old, "1-GPU-only (pre-fix run)", 0.45)):
        for prec, ls in (("float32", "-"), ("float64", ":")):
            rows = read_csv_rows(Path(src) / f"cs_{prec}" / "strong_scaling.csv")
            if not rows:
                continue
            by_res = defaultdict(list)
            for r in rows:
                by_res[int(r["resolution"])].append(
                    (int(r["n_gpus"]), float(r["time_per_step_ms"])))
            for res, pts in sorted(by_res.items()):
                pts = sorted(pts)
                if tag.startswith("1-GPU"):
                    pts = [p for p in pts if p[0] == 1]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax_a.plot(xs, ys, ls, marker="o", alpha=alpha,
                          label=f"C{res} {prec} ({tag})")
    ax_a.set_xscale("log", base=2); ax_a.set_yscale("log")
    ax_a.set_xticks([1, 2]); ax_a.set_xticklabels(["1", "2"])
    ax_a.set_xlabel("GPUs (RTX 8000)"); ax_a.set_ylabel("ms / step")
    ax_a.set_title("A  atm GPU strong — cubed-sphere dry JW")
    ax_a.grid(True, which="both", alpha=0.25); ax_a.legend(fontsize=6)

    # --- B: atm GPU weak --------------------------------------------------
    for prec, ls in (("float32", "-"), ("float64", ":")):
        rows = read_csv_rows(Path(args.drybaro) / f"cs_{prec}" / "weak_scaling.csv")
        if not rows:
            continue
        pts = sorted((int(r["n_gpus"]), float(r["time_per_step_ms"]),
                      float(r["cells_per_gpu"])) for r in rows)
        if not pts:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax_b.plot(xs, ys, ls, marker="s", label=f"{prec} (~{pts[0][2]:.0f} cells/GPU)")
    ax_b.set_xscale("log", base=2); ax_b.set_yscale("log")
    ax_b.set_xticks([1, 2]); ax_b.set_xticklabels(["1", "2"])
    ax_b.set_xlabel("GPUs"); ax_b.set_ylabel("ms / step")
    ax_b.set_title("B  atm GPU weak — fixed cells/GPU (comm cost visible)")
    ax_b.grid(True, which="both", alpha=0.25); ax_b.legend(fontsize=7)

    # --- C: atm CPU MPI ico ----------------------------------------------
    cmap = plt.get_cmap("tab10")
    for mode, marker in (("strong", "o"), ("weak", "^")):
        root = Path(args.cpumpi) / f"icosahedral_held_suarez_{mode}"
        series = defaultdict(list)
        for p in sorted(root.glob("*.json")):
            d = json.loads(p.read_text())
            key = (d["precision"], d["resolution"] if mode == "strong" else "weak")
            series[key].append((d["n_ranks"], d["sypd"]))
        for i, (key, pts) in enumerate(sorted(series.items())):
            prec, res = key
            pts = sorted(pts)
            xs, ys = zip(*pts)
            ls = "-" if prec == "float32" else ":"
            lbl = (f"I{res} {prec}" if mode == "strong" else f"weak {prec}")
            ax_c.plot(xs, ys, ls, marker=marker, color=cmap(i % 10), label=lbl)
    # ideal strong-scaling guide from I5 float32 np=1 if present
    ax_c.set_xscale("log", base=2); ax_c.set_yscale("log")
    ax_c.set_xlabel("MPI ranks (1 core each)"); ax_c.set_ylabel("SYPD")
    ax_c.set_title("C  atm CPU-MPI — icosahedral held_suarez (strong: fixed res; weak: auto)")
    ax_c.grid(True, which="both", alpha=0.25); ax_c.legend(fontsize=6, ncol=2)

    # --- D: ocean ---------------------------------------------------------
    for prec, ls in (("float32", "-"), ("float64", ":")):
        rows = read_csv_rows(Path(args.ocean_gpu) / prec / "ocean_scaling.csv")
        ll = [(float(r["total_cells"]), float(r["mcells_per_s"]))
              for r in rows if str(r.get("resolution", "")).startswith(("32", "64", "96", "128", "192"))
              and "LL" in r.get("mode", "") or True]
        # latlon rows: grid encoded in resolution column for this bench
        pts = sorted((float(r["total_cells"]), float(r["mcells_per_s"])) for r in rows)
        if pts:
            xs, ys = zip(*pts)
            ax_d.plot(xs, ys, ls, marker="o", label=f"GPU 1x RTX8000 {prec}")
    mrows = read_csv_rows(Path(args.ocean_mpi_csv))
    pts = sorted((int(r["n_gpus"]), float(r["mcells_per_s"]), r["mode"]) for r in mrows)
    if pts:
        for mode in {p[2] for p in pts}:
            sel = [(p[0], p[1]) for p in pts if p[2] == mode]
            xs, ys = zip(*sorted(sel))
            ax_d2 = ax_d  # same axes, secondary x meaning: annotate
            ax_d2.plot([x * 4e4 for x in xs], ys, "--s", alpha=0.7,
                       label=f"CPU-MPI {mode} (x=ranks*4e4 cells proxy)")
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.set_xlabel("total cells (incl. vertical)"); ax_d.set_ylabel("Mcells / s")
    ax_d.set_title("D  ocean — GPU size sweep (LL192 cliff) + CPU-MPI smoke")
    ax_d.grid(True, which="both", alpha=0.25); ax_d.legend(fontsize=6)

    fig.suptitle("legoESM scaling campaign — Ginsburg, 2026-06-10 (interim)", y=0.995)
    fig.text(0.01, 0.005,
             "A/B: post sharding+visibility fix (real 2-GPU). C: 24+ cases, sweep still running. "
             "D: cliff structural (persists across solver/advection variants). f64 RTX8000 = 1/32 FLOP rate; "
             "observed f32/f64 ~ 1.5-2.1x => bandwidth-bound.",
             fontsize=6.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
