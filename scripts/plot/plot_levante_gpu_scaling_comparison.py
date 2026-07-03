#!/usr/bin/env python
"""Combine per-GPU-count JSON results from run_levante_gpu_scaling.sh campaigns
into a single 4-panel comparison figure.

Outputs:
  scaling_comparison.png   — 4-panel: weak ms/step, strong SYPD, weak eff, strong eff
  scaling_summary.txt      — ASCII table of key results

Usage:
    python scripts/plot/plot_levante_gpu_scaling_comparison.py \\
        --campaign-dir /scratch/b/b309178/scaling/20260623T133921Z \\
        --output-dir   /scratch/b/b309178/scaling/20260623T133921Z

    # Multiple campaigns overlaid (label each):
    python scripts/plot/plot_levante_gpu_scaling_comparison.py \\
        --campaign-dir /scratch/.../20260623T133921Z \\
        --campaign-dir /scratch/.../20260624T080000Z \\
        --campaign-label run1 --campaign-label run2 \\
        --output-dir /scratch/.../combined
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScalingPoint:
    n_gpus: int
    resolution: int
    precision: str
    mode: str           # "weak" | "strong"
    time_per_step_ms: float
    sypd: float
    mcells_per_s: float
    scaling_efficiency: float
    campaign: str = ""


def _load_campaign(campaign_dir: Path, label: str = "") -> list[ScalingPoint]:
    """Load all scaling JSON files from a campaign directory tree."""
    points: list[ScalingPoint] = []
    cam_label = label or campaign_dir.name
    for mode in ("weak", "strong"):
        for json_path in sorted(campaign_dir.glob(f"gpu_*/*/{mode}_scaling.json")):
            try:
                data = json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for r in data.get("results", []):
                points.append(ScalingPoint(
                    n_gpus=int(r["n_gpus"]),
                    resolution=int(r["resolution"]),
                    precision=str(r["precision"]),
                    mode=mode,
                    time_per_step_ms=float(r["time_per_step_ms"]),
                    sypd=float(r["sypd"]),
                    mcells_per_s=float(r["mcells_per_s"]),
                    scaling_efficiency=float(r.get("scaling_efficiency", 1.0)),
                    campaign=cam_label,
                ))
    return points


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PREC_COLOR = {"float32": "#1565C0", "float64": "#C62828"}
_PREC_MARKER = {"float32": "o", "float64": "s"}
_PREC_LS = {"float32": "-", "float64": "--"}
_RES_PALETTE = ["#2E7D32", "#1565C0", "#6A1B9A", "#E65100", "#00695C", "#4E342E"]


def _res_colors(resolutions: list[int]) -> dict[int, str]:
    return {r: _RES_PALETTE[i % len(_RES_PALETTE)] for i, r in enumerate(sorted(resolutions))}


def make_figure(
    points: list[ScalingPoint],
    output_dir: Path,
) -> Path | None:
    """Create scaling_comparison.png in output_dir. Returns the path or None."""
    if not points:
        print("  No scaling data found — skipping plot.")
        return None

    weak = [p for p in points if p.mode == "weak"]
    strong = [p for p in points if p.mode == "strong"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("legoESM GPU Scaling (Levante A100) — Icosahedral Dycore",
                 fontsize=14, fontweight="bold")
    ax_wms, ax_sypd, ax_weff, ax_seff = axes.flat

    all_weak_gpus = sorted(set(p.n_gpus for p in weak))
    all_strong_gpus = sorted(set(p.n_gpus for p in strong))
    strong_resolutions = sorted(set(p.resolution for p in strong))
    res_colors = _res_colors(strong_resolutions)

    # --- Panel 1: Weak scaling ms/step ---
    ax = ax_wms
    for prec in ("float32", "float64"):
        pts = sorted([p for p in weak if p.precision == prec], key=lambda p: p.n_gpus)
        if not pts:
            continue
        gpus = [p.n_gpus for p in pts]
        ms = [p.time_per_step_ms for p in pts]
        ax.plot(gpus, ms, color=_PREC_COLOR[prec], marker=_PREC_MARKER[prec],
                ls=_PREC_LS[prec], lw=2, ms=7, label=prec, zorder=5)
        # Ideal: constant ms/step
        if gpus:
            ax.axhline(ms[0], color=_PREC_COLOR[prec], ls=":", alpha=0.35, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPUs", fontsize=11)
    ax.set_ylabel("ms / step", fontsize=11)
    ax.set_title("Weak Scaling — time per step", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    if all_weak_gpus:
        ax.set_xticks(all_weak_gpus)
        ax.set_xticklabels([str(g) for g in all_weak_gpus])

    # --- Panel 2: Strong scaling SYPD ---
    ax = ax_sypd
    for res in strong_resolutions:
        for prec in ("float32", "float64"):
            pts = sorted([p for p in strong if p.resolution == res and p.precision == prec],
                         key=lambda p: p.n_gpus)
            if not pts:
                continue
            gpus = [p.n_gpus for p in pts]
            sypd = [p.sypd for p in pts]
            lbl = f"I{res} {prec}"
            ax.plot(gpus, sypd, color=res_colors[res], marker=_PREC_MARKER[prec],
                    ls=_PREC_LS[prec], lw=2, ms=7, label=lbl, zorder=5)
            # Ideal linear speedup from 1-GPU baseline
            if len(gpus) > 1:
                base_sypd = sypd[0]
                base_gpus = gpus[0]
                gpu_arr = np.array(sorted(set(p.n_gpus for p in strong if p.resolution == res)))
                ax.plot(gpu_arr, base_sypd * (gpu_arr / base_gpus),
                        color=res_colors[res], ls=":", alpha=0.3, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("GPUs", fontsize=11)
    ax.set_ylabel("SYPD", fontsize=11)
    ax.set_title("Strong Scaling — SYPD", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    if all_strong_gpus:
        ax.set_xticks(all_strong_gpus)
        ax.set_xticklabels([str(g) for g in all_strong_gpus])

    # --- Panel 3: Weak scaling efficiency ---
    ax = ax_weff
    for prec in ("float32", "float64"):
        pts = sorted([p for p in weak if p.precision == prec], key=lambda p: p.n_gpus)
        if not pts:
            continue
        baseline_ms = pts[0].time_per_step_ms
        gpus = [p.n_gpus for p in pts]
        eff = [100.0 * baseline_ms / p.time_per_step_ms for p in pts]
        ax.plot(gpus, eff, color=_PREC_COLOR[prec], marker=_PREC_MARKER[prec],
                ls=_PREC_LS[prec], lw=2, ms=7, label=prec, zorder=5)
    ax.axhline(100.0, color="#555", ls=":", lw=1.5, alpha=0.5, label="ideal 100%")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 120)
    ax.set_xlabel("GPUs", fontsize=11)
    ax.set_ylabel("Efficiency (%)", fontsize=11)
    ax.set_title("Weak Scaling Efficiency", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    if all_weak_gpus:
        ax.set_xticks(all_weak_gpus)
        ax.set_xticklabels([str(g) for g in all_weak_gpus])

    # --- Panel 4: Strong scaling efficiency ---
    ax = ax_seff
    for res in strong_resolutions:
        for prec in ("float32", "float64"):
            pts = sorted([p for p in strong if p.resolution == res and p.precision == prec],
                         key=lambda p: p.n_gpus)
            if not pts:
                continue
            base_sypd = pts[0].sypd
            base_gpus = pts[0].n_gpus
            gpus = [p.n_gpus for p in pts]
            eff = [100.0 * p.sypd / (base_sypd * p.n_gpus / base_gpus) for p in pts]
            lbl = f"I{res} {prec}"
            ax.plot(gpus, eff, color=res_colors[res], marker=_PREC_MARKER[prec],
                    ls=_PREC_LS[prec], lw=2, ms=7, label=lbl, zorder=5)
    ax.axhline(100.0, color="#555", ls=":", lw=1.5, alpha=0.5, label="ideal 100%")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 120)
    ax.set_xlabel("GPUs", fontsize=11)
    ax.set_ylabel("Efficiency (%)", fontsize=11)
    ax.set_title("Strong Scaling Efficiency", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    if all_strong_gpus:
        ax.set_xticks(all_strong_gpus)
        ax.set_xticklabels([str(g) for g in all_strong_gpus])

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scaling_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out_path}")
    return out_path


def write_summary(points: list[ScalingPoint], output_dir: Path) -> Path:
    """Write a plain-text summary table."""
    lines = ["legoESM GPU Scaling Summary", "=" * 72, ""]

    for mode, header in (
        ("weak",   "Weak Scaling (constant cells/GPU, base level I4)"),
        ("strong", "Strong Scaling (fixed problem, increasing GPUs)"),
    ):
        subset = sorted(
            [p for p in points if p.mode == mode],
            key=lambda p: (p.precision, p.resolution, p.n_gpus),
        )
        if not subset:
            continue
        lines.append(header)
        lines.append("-" * 72)
        lines.append(f"  {'Prec':<8} {'Res':<5} {'GPUs':>6} {'ms/step':>10} "
                     f"{'SYPD':>10} {'Mcells/s':>10} {'Eff%':>7}")
        lines.append("  " + "-" * 60)
        for p in subset:
            lines.append(
                f"  {p.precision:<8} I{p.resolution:<4} {p.n_gpus:>6} "
                f"{p.time_per_step_ms:>10.3f} {p.sypd:>10.1f} "
                f"{p.mcells_per_s:>10.1f} {p.scaling_efficiency * 100:>7.1f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scaling_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"  Summary: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine GPU-count JSON results into a scaling comparison figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--campaign-dir", dest="campaign_dirs", action="append", default=[],
        metavar="DIR",
        help="Campaign directory (contains gpu_*/ sub-dirs). Repeat for overlays.",
    )
    p.add_argument(
        "--campaign-label", dest="campaign_labels", action="append", default=[],
        metavar="LABEL",
        help="Short label for each --campaign-dir (positional match).",
    )
    p.add_argument(
        "--output-dir", default=".",
        help="Directory to write scaling_comparison.png and scaling_summary.txt.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    dirs = [Path(d) for d in args.campaign_dirs]
    labels = args.campaign_labels + [""] * max(0, len(dirs) - len(args.campaign_labels))

    if not dirs:
        print("ERROR: provide at least one --campaign-dir")
        return 1

    all_points: list[ScalingPoint] = []
    for d, lbl in zip(dirs, labels):
        pts = _load_campaign(d, lbl)
        if not pts:
            print(f"  WARNING: no JSON results found under {d}")
        else:
            print(f"  Loaded {len(pts)} points from {d.name}")
        all_points.extend(pts)

    output_dir = Path(args.output_dir)
    make_figure(all_points, output_dir)
    write_summary(all_points, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
