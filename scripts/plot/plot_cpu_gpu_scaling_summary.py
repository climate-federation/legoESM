"""Consolidated weak+strong x CPU+GPU x atmosphere+ocean scaling summary.

The campaign's headline view in ONE figure: parallel efficiency for both
scaling modes (strong, weak), both backends (CPU multi-node MPI, GPU on-node
2-device), both components (atmosphere, ocean).  2x2 panels [component x mode];
within a panel, CPU runs are efficiency-vs-node CURVES (lines + circles) and the
GPU runs are 2-device POINTS (squares at N=2).

Two distinct data sources, joined here (NOT re-measured):
  * CPU curves   <- the tidy multi-node CSV (``aggregate_bcw_scaling`` ->
    ``multinode_clean/clean_tidy.csv``); efficiency normalised to each curve's
    own smallest device count via the SHARED ``efficiency_curve`` /``_group``
    core in ``plot_scaling_efficiency.py`` (no re-derived normalisation).
  * GPU points   <- the append-only ledger ``docs/scaling/scaling_indicators.csv``
    rows whose metric is an ``eff_2gpu_*`` (full-step, real RTX8000 PCIe pair)
    or an ``ocean_spmd_pcg_*_eff_ndev2`` (the barotropic-PCG KERNEL proxy, a
    comm-bound lower bound -- plotted with a distinct hollow marker and NEVER
    conflated with the full-step number, per the campaign honesty rule).

HONEST-AXIS NOTE: the x-axis mixes CPU "nodes/ranks" and GPU "devices on one
PCIe pair" -- it is a parallel-unit count, and every series is normalised to its
OWN baseline, so the efficiencies are comparable as comm-vs-compute ratios while
the absolute unit differs (stated in the axis label + subtitle).  This mirrors
the existing ``plot_scaling_efficiency.py`` convention (BACK_MK CPU=o / GPU=s).

    python scripts/plot/plot_cpu_gpu_scaling_summary.py \\
        --tidy results/scaling_ginsburg/multinode_clean/clean_tidy.csv \\
        --indicators docs/scaling/scaling_indicators.csv \\
        --out docs/scaling
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the SHARED efficiency core (no copied normalisation numerics).
_EFF_SPEC = Path(__file__).resolve().parent / "plot_scaling_efficiency.py"
_spec = importlib.util.spec_from_file_location("plot_scaling_efficiency", _EFF_SPEC)
_eff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eff)

GRID_COLOR = _eff.GRID_COLOR          # grid -> colour (shared palette)
PANELS = _eff.PANELS                  # [(component, mode), ...] (atm/ocean x strong/weak)

# grid token in scaling_indicators.csv -> (component, tidy grid name)
_GRID_MAP = {
    "atm_cube": ("atm", "cubed-sphere"),
    "atm_latlon": ("atm", "latlon"),
    "atm_icosahedral": ("atm", "icosahedral"),
    "ocean_latlon": ("ocean", "latlon"),
}


def _prec_from_metric(metric: str) -> str:
    """f32/f64 hint embedded in the metric name; default f64 (campaign default)."""
    if "f32" in metric or "float32" in metric:
        return "float32"
    return "float64"


def gpu_points(indicator_rows: list[dict]) -> dict:
    """(component, mode) -> list of GPU 2-device efficiency points.

    Each point: dict(comp, mode, grid, precision, eff, kind, label, job).
    ``kind`` is ``"fullstep"`` (real full-model 2-GPU efficiency, metric
    ``eff_2gpu_*``) or ``"kernel"`` (the ocean barotropic-PCG SPMD comm-bound
    proxy, metric EXACTLY ``ocean_spmd_pcg_*_eff_ndev2`` on the ocean grid -- a
    LOWER bound drawn hollow, never merged with the full-step value).  All other
    rows are ignored.  A row whose ``mode`` is neither strong nor weak is
    SKIPPED (never silently assumed strong).  The ledger is append-only, so a
    later row with the same ``(grid, metric)`` SUPERSEDES the earlier one
    (last-wins de-dup) -- otherwise reruns/corrections double-plot.
    """
    latest: dict = {}            # (gtok, metric) -> parsed point (last wins)
    order: list = []             # preserve first-seen order for stable output
    for r in indicator_rows:
        metric = (r.get("metric") or "").strip()
        is_fullstep = metric.startswith("eff_2gpu_")
        is_kernel = (metric.startswith("ocean_spmd_pcg_")
                     and metric.endswith("_eff_ndev2"))
        if not (is_fullstep or is_kernel):
            continue
        gtok = (r.get("grid") or "").strip()
        if gtok not in _GRID_MAP:
            continue
        if is_kernel and gtok != "ocean_latlon":
            continue             # the baro-PCG kernel proxy is ocean-only
        try:
            eff = float(r["value"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (eff > 0.0):
            continue
        mode_raw = (r.get("mode") or "").strip()
        if mode_raw.startswith("weak"):
            mode = "weak"
        elif mode_raw.startswith("strong"):
            mode = "strong"
        else:
            continue             # unknown/blank mode -> skip, never assume strong
        comp, grid = _GRID_MAP[gtok]
        kind = "kernel" if is_kernel else "fullstep"
        prec = _prec_from_metric(metric)
        label = f"{grid} {prec[-2:]} GPU" + (
            " (baro-PCG kernel)" if kind == "kernel" else "")
        key = (gtok, metric)
        if key not in latest:
            order.append(key)
        latest[key] = dict(comp=comp, mode=mode, grid=grid, precision=prec,
                           eff=eff, kind=kind, label=label,
                           job=(r.get("job") or "").strip())
    out: dict = {}
    for key in order:
        p = latest[key]
        out.setdefault((p["comp"], p["mode"]), []).append(p)
    return out


def make_summary_figure(tidy_rows: list[dict], indicator_rows: list[dict],
                        out_dir: Path) -> Path:
    cpu = _eff._group(tidy_rows)            # (comp,mode) -> (grid,prec,back) -> {n: mc}
    gpu = gpu_points(indicator_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (comp, mode) in zip(axes.flat, PANELS):
        drew = False
        max_y = 1.0
        # --- CPU multi-node efficiency curves (lines + circles) ---
        for (grid, prec, backend), nd_to_mc in sorted(cpu.get((comp, mode), {}).items()):
            pts = _eff.efficiency_curve(nd_to_mc)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            max_y = max(max_y, max(ys))
            ax.plot(xs, ys, marker="o", ms=5,
                    color=GRID_COLOR.get(grid, "#777777"),
                    linestyle=("-" if prec == "float64" else "--"),
                    alpha=0.9, label=f"{grid} {prec[-2:]} CPU")
            drew = True
        # --- GPU 2-device efficiency points (squares; hollow=kernel proxy) ---
        for p in sorted(gpu.get((comp, mode), []), key=lambda d: (d["grid"], d["kind"])):
            color = GRID_COLOR.get(p["grid"], "#777777")
            hollow = p["kind"] == "kernel"
            ax.plot([2], [p["eff"]], marker="s", ms=11,
                    color=color,
                    markerfacecolor=("none" if hollow else color),
                    markeredgecolor=color, markeredgewidth=2.0,
                    linestyle="None", label=p["label"])
            ax.annotate(f"{p['eff']:.2f}", (2, p["eff"]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=7, color=color, va="center")
            max_y = max(max_y, p["eff"])
            drew = True
        if drew:
            ax.axhline(1.0, color="k", ls="--", alpha=0.5, lw=1, label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32])
            ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
            ax.set_ylim(0.0, max(1.25, 1.05 * max_y))
            ax.legend(fontsize=7, ncol=2, loc="lower left")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(f"{comp.upper()} — {mode} scaling")
        ax.set_xlabel("parallel units  (CPU: nodes/ranks · GPU: devices, RTX8000 PCIe pair)")
        ax.set_ylabel("parallel efficiency  (1 = ideal)")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("legoESM weak+strong scaling — CPU multi-node (lines, ○) vs "
                 "2-GPU on-node (squares, □; hollow □ = barotropic-PCG kernel proxy)\n"
                 "f64 solid / f32 dashed · efficiency normalised per series · 1.0 = ideal",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cpu_gpu_scaling_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tidy",
                   default="results/scaling_ginsburg/multinode_clean/clean_tidy.csv")
    p.add_argument("--indicators", default="docs/scaling/scaling_indicators.csv")
    p.add_argument("--out", default="docs/scaling")
    args = p.parse_args()
    tidy_rows = _eff._read(Path(args.tidy))
    indicator_rows = _eff._read(Path(args.indicators))
    out = make_summary_figure(tidy_rows, indicator_rows, Path(args.out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
