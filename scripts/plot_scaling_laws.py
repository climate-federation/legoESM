"""Plot strong and weak scaling laws for the GPU-scaling branch results.

Reads the ``strong_scaling.csv`` and ``weak_scaling.csv`` files produced
by ``scripts/run_levante_gpu_scaling.py`` and plots:

1. Strong scaling: time-per-step (and SYPD) vs n_gpus, with the ideal
   ``T_1 / n_gpus`` reference and per-grid efficiency.
2. Weak scaling: time-per-step vs n_gpus at fixed cells/GPU, with the
   ideal flat reference.
3. SPMD-vs-iter improvement chart for cubed-sphere C24 strong scaling
   (iter-49 multi-face SPMD + iter-58 merged halo vs the iter-46 baseline).

Output directory defaults to ``results/scaling_plots/``.

Usage
-----
    python scripts/plot_scaling_laws.py
    python scripts/plot_scaling_laws.py --output-dir results/my_plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple


class StrongRow(NamedTuple):
    n_gpus: int
    resolution: int
    n_levels: int
    precision: str
    time_per_step_ms: float
    sypd: float
    scaling_efficiency: float


def _read_strong(csv_path: Path) -> list[StrongRow]:
    rows: list[StrongRow] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                StrongRow(
                    n_gpus=int(r["n_gpus"]),
                    resolution=int(r["resolution"]),
                    n_levels=int(r["n_levels"]),
                    precision=r["precision"],
                    time_per_step_ms=float(r["time_per_step_ms"]),
                    sypd=float(r["sypd"]),
                    scaling_efficiency=float(r["scaling_efficiency"]),
                )
            )
    return rows


def _try_imports():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc


def plot_strong(plt, datasets: dict[str, list[StrongRow]], output_dir: Path) -> None:
    """Strong scaling: time/step + efficiency vs n_gpus per grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        "cubed-sphere C24": "#2196F3",
        "cubed-sphere C48": "#1565C0",
        "spectral T21": "#F44336",
        "icosahedral L4": "#4CAF50",
    }

    for label, rows in datasets.items():
        rows_sorted = sorted(rows, key=lambda r: r.n_gpus)
        n_gpus = [r.n_gpus for r in rows_sorted]
        ms = [r.time_per_step_ms for r in rows_sorted]
        eff = [r.scaling_efficiency for r in rows_sorted]
        c = colors.get(label, "#666666")
        ax1.plot(n_gpus, ms, "o-", color=c, label=label, linewidth=2, markersize=7)
        ax2.plot(n_gpus, eff, "o-", color=c, label=label, linewidth=2, markersize=7)

    # Ideal strong-scaling reference (from each label's 1-device baseline).
    for label, rows in datasets.items():
        rows_sorted = sorted(rows, key=lambda r: r.n_gpus)
        if not rows_sorted:
            continue
        t1 = rows_sorted[0].time_per_step_ms
        ng = [r.n_gpus for r in rows_sorted]
        ideal = [t1 / n for n in ng]
        c = colors.get(label, "#666666")
        ax1.plot(ng, ideal, "--", color=c, alpha=0.4)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("n_gpus")
    ax1.set_ylabel("time/step (ms)")
    ax1.set_title("Strong scaling — time per step\n(dashed = ideal $T_1/n$)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.axhline(1.0, linestyle=":", color="black", alpha=0.5, label="ideal (1.0)")
    ax2.set_xscale("log")
    ax2.set_xlabel("n_gpus")
    ax2.set_ylabel("scaling efficiency")
    ax2.set_title("Strong scaling — efficiency\n($T_1 / (n \\cdot T_n)$)")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Strong scaling on emulated multi-CPU "
        "(JAX shard_map, XLA_FLAGS=--xla_force_host_platform_device_count=N)\n"
        "Note: emulated devices share physical cores — peak speedup limited by core count.",
        fontsize=10,
    )
    fig.tight_layout()
    out = output_dir / "strong_scaling.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_weak(plt, datasets: dict[str, list[StrongRow]], output_dir: Path) -> None:
    """Weak scaling: time/step vs n_gpus at fixed cells/GPU."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {
        "cubed-sphere": "#2196F3",
        "spectral": "#F44336",
        "icosahedral": "#4CAF50",
    }

    if not datasets:
        ax.text(
            0.5, 0.5, "No weak-scaling CSVs found", ha="center", va="center",
            transform=ax.transAxes,
        )
        out = output_dir / "weak_scaling.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out} (empty)")
        return

    for label, rows in datasets.items():
        rows_sorted = sorted(rows, key=lambda r: r.n_gpus)
        n_gpus = [r.n_gpus for r in rows_sorted]
        ms = [r.time_per_step_ms for r in rows_sorted]
        c = colors.get(label.split()[0].lower(), "#666666")
        ax.plot(n_gpus, ms, "o-", color=c, label=label, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("n_gpus")
    ax.set_ylabel("time/step (ms)")
    ax.set_title(
        "Weak scaling — fixed cells/GPU\n"
        "(ideal: flat curve)",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = output_dir / "weak_scaling.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_iter_progression(plt, output_dir: Path) -> None:
    """Compare iter-46 baseline vs iter-49 vs iter-58 cubed-sphere C24 strong."""
    paths = {
        "iter-46 (single-face SPMD halo + cumsum reuse)":
            Path("results/scaling_iter46/cs"),
        "iter-49 (multi-face SPMD halo at 2/3 dev)":
            Path("results/scaling_iter49/cs"),
        "iter-58 (merged cell-field halo)":
            Path("results/scaling_iter58/cs"),
    }

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    iter_colors = {
        "iter-46 (single-face SPMD halo + cumsum reuse)": "#999999",
        "iter-49 (multi-face SPMD halo at 2/3 dev)": "#FF9800",
        "iter-58 (merged cell-field halo)": "#2196F3",
    }

    found_any = False
    for label, base in paths.items():
        if not base.exists():
            continue
        # Find latest timestamped subdir
        subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
        if not subdirs:
            continue
        csv_path = subdirs[-1] / "strong_scaling.csv"
        if not csv_path.exists():
            continue
        rows = _read_strong(csv_path)
        rows_sorted = sorted(rows, key=lambda r: r.n_gpus)
        ng = [r.n_gpus for r in rows_sorted]
        ms = [r.time_per_step_ms for r in rows_sorted]
        c = iter_colors[label]
        ax.plot(ng, ms, "o-", color=c, label=label, linewidth=2, markersize=8)
        found_any = True

    # Ideal reference using the iter-58 1-device baseline if available.
    for label in ("iter-58 (merged cell-field halo)", "iter-49 (multi-face SPMD halo at 2/3 dev)"):
        base = paths.get(label)
        if base is None or not base.exists():
            continue
        subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
        if not subdirs:
            continue
        csv_path = subdirs[-1] / "strong_scaling.csv"
        if not csv_path.exists():
            continue
        rows = sorted(_read_strong(csv_path), key=lambda r: r.n_gpus)
        if rows:
            t1 = rows[0].time_per_step_ms
            ng = [r.n_gpus for r in rows]
            ideal = [t1 / n for n in ng]
            ax.plot(ng, ideal, "--", color="black", alpha=0.4, label="ideal $T_1/n$")
            break

    if not found_any:
        ax.text(
            0.5, 0.5, "No iter-* CSVs found", ha="center", va="center",
            transform=ax.transAxes,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_gpus")
    ax.set_ylabel("time/step (ms)")
    ax.set_title(
        "Cubed-sphere C24 strong scaling — iteration progression\n"
        "(iter-46 baseline → iter-49 multi-face SPMD → iter-58 merged halo)",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = output_dir / "iter_progression.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _latest_csv(base: Path, name: str = "strong_scaling.csv") -> Path | None:
    if not base.exists():
        return None
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()])
    for d in reversed(subdirs):
        p = d / name
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/scaling_plots"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt = _try_imports()

    # ------------------- strong scaling datasets --------------------
    strong_paths = {
        "cubed-sphere C24": _latest_csv(Path("results/scaling_iter58/cs")),
        "cubed-sphere C48": _latest_csv(Path("results/scaling_iter49/cs_c48")),
        "spectral T21": _latest_csv(Path("results/scaling_iter49/spec")),
        "icosahedral L4": _latest_csv(Path("results/scaling_iter49/ico")),
    }
    strong_datasets: dict[str, list[StrongRow]] = {}
    for label, p in strong_paths.items():
        if p is None:
            print(f"  no CSV for {label}")
            continue
        strong_datasets[label] = _read_strong(p)
        print(f"  loaded {label}: {p}")

    if strong_datasets:
        plot_strong(plt, strong_datasets, args.output_dir)
    else:
        print("  no strong-scaling data — skipping strong plot")

    # ------------------- weak scaling datasets ----------------------
    weak_paths = {
        "cubed-sphere": _latest_csv(Path("results/scaling_iter63/cs"), "weak_scaling.csv"),
        "icosahedral": _latest_csv(Path("results/scaling_iter63/ico"), "weak_scaling.csv"),
        "spectral": _latest_csv(Path("results/scaling_iter63/spec"), "weak_scaling.csv"),
    }
    weak_datasets: dict[str, list[StrongRow]] = {}
    for label, p in weak_paths.items():
        if p is None:
            print(f"  no weak CSV for {label}")
            continue
        weak_datasets[label] = _read_strong(p)
        print(f"  loaded weak {label}: {p}")

    plot_weak(plt, weak_datasets, args.output_dir)

    # ------------------- iter progression ---------------------------
    plot_iter_progression(plt, args.output_dir)


if __name__ == "__main__":
    main()
