"""Prior-format 3D-atmosphere strong-scaling figure, f32 (thick) vs f64 (thin).

Reuses the validated prior ladder2n f32 receipts ({series}_n*.jsonl) as the
THICK curves and overlays the genuine f64 arms ({series}_f64_n*.jsonl, from
strong_ladder_prior_f64.sbatch) as THIN curves in the same colour. Large meshes
only (small/coarse meshes cannot strong-scale and are excluded). Panels: ms/step
(log-log, dashed = ideal from each curve's smallest point) + parallel efficiency.

    python scripts/plot/plot_ladder_2n_precision.py \
        --receipts /scratch/b/b381103/legoesm_scaling/ladder2n --out ladder2n_prec.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, f32 globs, f64 globs, colour, marker)
SERIES = [
    ("MPAS L9 sfc", ("mpas_n*.jsonl", "mpas_s9_n*.jsonl"), ("mpas_s9_f64_n*.jsonl",), "#0072B2", "o"),
    ("MPAS L9 metis", ("mpas_s9_metis_n*.jsonl",), ("mpas_s9_metis_f64_n*.jsonl",), "#009E73", "v"),
    ("MPAS L10", ("mpas_s10_n*.jsonl",), ("mpas_s10_f64_n*.jsonl",), "#56B4E9", "s"),
    ("lat-lon 2048x4096", ("latlon_n*.jsonl",), ("latlon_f64_n*.jsonl",), "#D55E00", "^"),
    ("lat-lon 4096x8192", ("latlon_4096_n*.jsonl",), ("latlon_4096_f64_n*.jsonl",), "#E69F00", "D"),
]
MACHINE_MAX_POW2 = 128


def _size(rec):
    nlev = rec.get("nlev") or rec.get("n_levels")
    if rec.get("n_cells") and nlev:
        return f"{rec['n_cells']:,} cells x {nlev}L"
    if rec.get("n_lat") and rec.get("n_lon") and nlev:
        return f"{rec['n_lat']}x{rec['n_lon']} x {nlev}L"
    return None


def load(receipt_dir, patterns):
    best, size = {}, None
    for pattern in patterns:
        for path in glob.glob(os.path.join(receipt_dir, pattern)):
            lines = [ln for ln in open(path) if ln.strip()]
            if not lines:
                continue
            rec = json.loads(lines[-1])
            ms, n = rec.get("steady_median_ms"), rec.get("n_devices")
            if ms is None or n is None:
                continue
            job = rec.get("metadata", {}).get("slurm_job_id")
            key = int(job) if job and str(job).isdigit() else -1
            if n not in best or key > best[n][0]:
                best[n] = (key, float(ms))
            size = size or _size(rec)
    return sorted((n, v[1]) for n, v in best.items()), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", default="/scratch/b/b381103/legoesm_scaling/ladder2n")
    ap.add_argument("--out", default="ladder2n_prec.png")
    args = ap.parse_args()

    fig, (ax, axe) = plt.subplots(1, 2, figsize=(14, 6))
    any_f64 = False

    for label, f32_globs, f64_globs, color, marker in SERIES:
        for globs, lw, tag in ((f32_globs, 3.0, "f32"), (f64_globs, 1.3, "f64")):
            pts, size = load(args.receipts, globs)
            if len(pts) < 2:
                if tag == "f32":
                    print(f"[skip] {label} {tag}: <2 points")
                continue
            if tag == "f64":
                any_f64 = True
            n = [p[0] for p in pts]
            ms = [p[1] for p in pts]
            lbl = f"{label} ({size})" if (size and tag == "f32") else None
            ax.plot(n, ms, marker=marker, color=color, lw=lw, ms=6, label=lbl,
                    markerfacecolor=("none" if tag == "f64" else color))
            n0, t0 = n[0], ms[0]
            ax.plot(n, [t0 * n0 / k for k in n], ls="--", color=color, lw=0.8, alpha=0.4)
            eff = [100.0 * (t0 * n0 / k) / t for k, t in zip(n, ms)]
            axe.plot(n, eff, marker=marker, color=color, lw=lw, ms=6,
                     markerfacecolor=("none" if tag == "f64" else color))
            print(f"{label} {tag}: " + "  ".join(f"{k}g {t:.2f}ms" for k, t in pts))

    for a in (ax, axe):
        a.set_xscale("log", base=2)
        a.set_xlabel("GPUs (A100, Levante)")
        a.axvline(252, color="0.5", ls="-.", lw=1.0)
        a.grid(True, which="both", alpha=0.25)
    ax.set_yscale("log")
    ax.set_ylabel("ms / step")
    ax.set_title("Strong scaling, 3D atmosphere (dashed = ideal)")
    ax.legend(fontsize=8, loc="lower left")
    axe.axhline(100.0, color="0.4", ls=":", lw=1.0)
    axe.set_ylabel("parallel efficiency (%)")
    axe.set_ylim(0, 110)
    axe.set_title("Parallel efficiency vs smallest device count")
    prec_handles = [
        Line2D([], [], color="#444", lw=3.0, label="f32 (thick)"),
        Line2D([], [], color="#444", lw=1.3, label="f64 (thin)"),
    ]
    axe.legend(handles=prec_handles, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  (f64 overlaid: {any_f64})")


if __name__ == "__main__":
    main()
