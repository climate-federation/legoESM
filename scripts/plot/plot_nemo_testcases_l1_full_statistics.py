#!/usr/bin/env python3
"""Plot the preregistered lane-1 full-duration statistical comparison.

This script does no integration and computes no new statistical metric. It
loads the committed-run legoESM states and the NEMO states through the full-
duration scorer, which in turn reuses the certified phase-3 halo, staggering,
and wet-mask implementation. The scorer JSON supplies every plotted metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = (
    REPO_ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_full_statistics.py"
)
STATS_SPEC = importlib.util.spec_from_file_location("nemo_l1_full_statistics", STATS_PATH)
if STATS_SPEC is None or STATS_SPEC.loader is None:
    raise ImportError(f"cannot load full-duration scorer at {STATS_PATH}")
STATS = importlib.util.module_from_spec(STATS_SPEC)
STATS_SPEC.loader.exec_module(STATS)

DEFAULT_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1")
DEFAULT_FULL_ROOT = DEFAULT_ROOT / "full_statistical"
DEFAULT_OUTPUT = DEFAULT_ROOT / "figures_full"
DEFAULT_COPY = Path("/tmp/l1_figures_full")
ARMS = ("N2", "N4", "L64", "L32")
ARM_LABELS = {
    "N2": "NEMO FCT2 oracle",
    "N4": "NEMO FCT4",
    "L64": "legoESM fp64",
    "L32": "legoESM fp32",
}
ARM_COLORS = {"N2": "#111111", "N4": "#7570b3", "L64": "#d95f02", "L32": "#1b9e77"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_colorbar(colorbar) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    colorbar.formatter = formatter
    colorbar.update_ticks()


@dataclass(frozen=True)
class SectionData:
    case: str
    row: int
    mask: np.ndarray
    h_partial: np.ndarray
    bathymetry_m: np.ndarray
    x_edges_km: np.ndarray
    times_s: tuple[int, ...]
    oracle: dict[int, np.ndarray]
    candidate: dict[int, np.ndarray]
    state_hash: str
    run_sha: str


def section_data(case: str, lego_root: Path) -> SectionData:
    spec = STATS.CASES[case]
    oracle = STATS.load_nemo_states(case, Path(spec["baseline"]), baseline=True)
    candidate, metadata = STATS.load_legoesm_states(case, "fp64", lego_root)
    require(tuple(oracle) == tuple(candidate), f"{case}: section time mismatch")
    card = STATS.build_nemo_testcase_card(case)
    mask3 = np.asarray(STATS.expected_masks(card)["T"], dtype=bool)
    counts = np.count_nonzero(mask3, axis=(1, 2))
    row = int(np.argmax(counts))
    require(np.count_nonzero(counts == counts[row]) == 1, f"{case}: ambiguous section row")
    dx = np.asarray(card.recipe.grid.dx_T, dtype=np.float64)[row]
    x_edges = np.concatenate(([0.0], np.cumsum(dx))) / 1000.0
    for time_s in oracle:
        require(
            oracle[time_s]["T"].shape == candidate[time_s]["T"].shape == mask3.shape,
            f"{case} {time_s}: section shape mismatch",
        )
    return SectionData(
        case=case,
        row=row,
        mask=mask3[row],
        h_partial=np.asarray(card.recipe.z_coord.h_partial, dtype=np.float64)[row],
        bathymetry_m=np.asarray(card.recipe.initial_state.H_bathy.data, dtype=np.float64)[row],
        x_edges_km=x_edges,
        times_s=tuple(oracle),
        oracle={time_s: values["T"][row] for time_s, values in oracle.items()},
        candidate={time_s: values["T"][row] for time_s, values in candidate.items()},
        state_hash=metadata["states_sha256"],
        run_sha=metadata["git_sha"],
    )


def polygons(data: SectionData) -> tuple[list, np.ndarray]:
    result = []
    indices = []
    for ix in range(data.mask.shape[0]):
        interfaces = np.concatenate(([0.0], np.cumsum(data.h_partial[ix])))
        for iz in np.flatnonzero(data.mask[ix]):
            x0, x1 = data.x_edges_km[ix : ix + 2]
            z0, z1 = interfaces[iz : iz + 2]
            require(z1 > z0, f"{data.case}: nonpositive active thickness")
            result.append([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
            indices.append((ix, iz))
    require(bool(result), f"{data.case}: no section cells")
    return result, np.asarray(indices, dtype=np.int64)


def add_section(ax, data, values, section_polygons, indices, cmap, norm):
    plotted = np.asarray(values)[indices[:, 0], indices[:, 1]]
    require(np.all(np.isfinite(plotted)), f"{data.case}: nonfinite section")
    collection = PolyCollection(
        section_polygons,
        array=plotted,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )
    ax.add_collection(collection)
    bottom = np.r_[data.bathymetry_m, data.bathymetry_m[-1]]
    depth_max = float(np.max(data.bathymetry_m))
    ax.fill_between(
        data.x_edges_km,
        bottom,
        depth_max * 1.025,
        step="post",
        color="0.78",
        zorder=3,
    )
    ax.step(data.x_edges_km, bottom, where="post", color="black", linewidth=0.9, zorder=4)
    ax.set_xlim(data.x_edges_km[0], data.x_edges_km[-1])
    ax.set_ylim(depth_max * 1.025, 0.0)
    return collection


def plot_sections(data: SectionData, output: Path, figure_sha: str) -> Path:
    section_polygons, indices = polygons(data)
    wet = np.concatenate(
        [
            values[data.mask]
            for time_s in data.times_s
            for values in (data.oracle[time_s], data.candidate[time_s])
        ]
    )
    state_norm = Normalize(float(np.min(wet)), float(np.max(wet)))
    fig, axes = plt.subplots(
        3, 3, figsize=(13.4, 8.4), sharex=True, sharey=True, constrained_layout=True
    )
    fig.get_layout_engine().set(rect=(0.0, 0.08, 1.0, 0.88))
    fig.suptitle(f"{data.case}: full-duration temperature sections", y=0.988)
    state_map = None
    for column, time_s in enumerate(data.times_s):
        oracle = data.oracle[time_s]
        candidate = data.candidate[time_s]
        difference = candidate - oracle
        limit = float(np.max(np.abs(difference[data.mask])))
        require(limit > 0.0 or time_s == 0, f"{data.case} {time_s}: vacuous difference")
        display_limit = limit if limit > 0.0 else 1.0e-15
        state_map = add_section(
            axes[0, column],
            data,
            oracle,
            section_polygons,
            indices,
            "inferno",
            state_norm,
        )
        add_section(
            axes[1, column],
            data,
            candidate,
            section_polygons,
            indices,
            "inferno",
            state_norm,
        )
        diff_map = add_section(
            axes[2, column],
            data,
            difference,
            section_polygons,
            indices,
            "RdBu_r",
            Normalize(-display_limit, display_limit),
        )
        completed = int(round(time_s / float(STATS.CASES[data.case]["dt_s"])))
        frame = (
            "initial Nbb/before"
            if time_s == 0
            else ("Nbb/before" if time_s != data.times_s[-1] else "final restart")
        )
        axes[0, column].set_title(f"completed={completed:,}; t={time_s / 3600:.3f} h\n{frame}")
        axes[2, column].set_title(f"legoESM − NEMO; own scale ±{limit:.3e} K", fontsize=9)
        colorbar = fig.colorbar(
            diff_map, ax=axes[2, column], orientation="horizontal", pad=0.15, fraction=0.08
        )
        scientific_colorbar(colorbar)
        colorbar.set_label("Temperature difference (K)")
    for index, label in enumerate(("NEMO FCT2", "legoESM fp64", "Difference")):
        axes[index, 0].set_ylabel(f"{label}\nDepth (m)")
    for ax in axes[-1]:
        ax.set_xlabel("x (km)")
    require(state_map is not None, "missing state color scale")
    colorbar = fig.colorbar(state_map, ax=axes[:2, :], location="right", fraction=0.025)
    colorbar.set_label("Temperature (°C)")
    footer = (
        f"case={data.case}; frame=certified Nbb entries at initial/midpoint and "
        "tn restart at final; "
        "T-centre index comparison; phase-3 halo stripping; common wet-T mask; no interpolation\n"
        f"topography=certified rest partial-cell profile; legoESM=loaded CPU fp64 artifact "
        f"sha256={data.state_hash[:16]}; run_git={data.run_sha}; figure_git={figure_sha}"
    )
    fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=6.5)
    path = output / f"{STATS.CASES[data.case]['slug']}_full_temperature_sections.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Description": footer})
    plt.close(fig)
    return path


def verdict_map(report: dict) -> dict[str, str]:
    return {row["name"]: row["verdict"] for row in report["rows"]}


def curve_panel(ax, report, key, ylabel, title, verdict_key=None):
    for arm in ARMS:
        metric = report["metrics"][arm]
        times_h = np.asarray(metric["times_s"], dtype=np.float64) / 3600.0
        values = np.asarray(metric[key], dtype=np.float64)
        ax.plot(
            times_h,
            values,
            marker="o",
            linewidth=1.7,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    verdict = verdict_map(report).get(verdict_key or key, "")
    ax.set_title(f"{title}\n{verdict}", fontsize=9.5)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", linewidth=0.6)


def bridge_panel(ax, report, field):
    series = report["deterministic_bridge"]["series"][field]
    steps = np.asarray(series["completed_steps"])
    values = np.asarray(series["normalized_max_abs"])
    positive = values > 0.0
    ax.plot(steps[positive], values[positive], color="#b2182b", marker="o", markersize=2.5)
    ax.set_xscale("symlog", linthresh=60)
    ax.set_yscale("log")
    ax.axvline(59, color="0.45", linestyle="--", linewidth=0.8)
    name = "temperature_linf" if field == "T" else "instantaneous_u_linf"
    ax.set_title(
        f"Deterministic {series['staggering']} bridge\n{verdict_map(report)[name]}", fontsize=9.5
    )
    ax.set_xlabel("Completed steps (dashed: end of kt60 ladder)")
    ax.set_ylabel(r"Normalized wet $L_\infty$")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)


def plot_metrics(case: str, report: dict, output: Path, figure_sha: str) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.09, 1.0, 0.81))
    fig.suptitle(f"{case}: preregistered full-duration metrics", y=0.99)
    if case == "OVERFLOW-zps":
        curve_panel(axes[0, 0], report, "plume_descent_m", "Depth (m)", "Deepest cold level")
        curve_panel(axes[0, 1], report, "plume_front_km", "x (km)", "Bottom plume front")
        bins = np.asarray(report["metrics"]["N2"]["final_temperature_bins_C"])
        centres = 0.5 * (bins[:-1] + bins[1:])
        for arm in ARMS:
            values = report["metrics"][arm]["final_temperature_histogram"]
            axes[0, 2].step(
                centres, values, where="mid", color=ARM_COLORS[arm], label=ARM_LABELS[arm]
            )
        axes[0, 2].set_title(
            "Final slope T distribution\n" + verdict_map(report)["final_temperature_histogram_tv"],
            fontsize=9.5,
        )
        axes[0, 2].set(xlabel="Temperature (°C)", ylabel="Volume probability")
        labels = report["metrics"]["N2"]["final_water_mass_labels"]
        width = 0.19
        x = np.arange(len(labels))
        for index, arm in enumerate(ARMS):
            axes[1, 0].bar(
                x + (index - 1.5) * width,
                report["metrics"][arm]["final_water_mass_census"],
                width,
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm],
            )
        axes[1, 0].set_xticks(x, labels)
        axes[1, 0].set_ylabel("Volume fraction")
        axes[1, 0].set_title(
            "Final slope water-mass census\n" + verdict_map(report)["final_water_mass_census"],
            fontsize=9.5,
        )
    else:
        curve_panel(axes[0, 0], report, "front_position_km", "x (km)", "Dense front position")
        curve_panel(
            axes[0, 1],
            report,
            "rpe_relative",
            "(RPE−RPE₀)/|RPE₀|",
            "Reference potential energy",
        )
        curve_panel(
            axes[0, 2],
            report,
            "temperature_variance_fraction",
            "Variance / initial variance",
            "Temperature variance decay",
        )
        ratios = [report["metrics"][arm]["front_speed_anchor_ratio"] for arm in ARMS]
        axes[1, 0].bar(ARMS, ratios, color=[ARM_COLORS[arm] for arm in ARMS])
        axes[1, 0].axhline(1.0, color="0.2", linestyle="--", linewidth=0.9)
        axes[1, 0].set_ylabel("fitted speed / Benjamin anchor")
        axes[1, 0].set_title(
            "Front speed versus analytic anchor\n"
            + verdict_map(report)["front_speed_anchor_ratio"],
            fontsize=9.5,
        )
    bridge_panel(axes[1, 1], report, "T")
    bridge_panel(axes[1, 2], report, "u")
    handles = [
        plt.Line2D([], [], color=ARM_COLORS[arm], marker="o", label=ARM_LABELS[arm]) for arm in ARMS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
    )
    footer = (
        f"case={case}; three registered states only; N2/N4/L64/L32; exact floor "
        "classes read from scorer; float64 reductions; deterministic bridge uses "
        "certified kt1..60 Nbb rows then midpoint Nbb and final restart\n"
        "mask/frame=phase-3 certified halo, T-centre and instantaneous-U common "
        "wet intersections; "
        f"scorer_prereg={report['preregistration_commit']}; figure_git={figure_sha}"
    )
    fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=6.5)
    path = output / f"{STATS.CASES[case]['slug']}_full_statistical_metrics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Description": footer})
    plt.close(fig)
    return path


def plot_failure(case: str, report: dict, output: Path, figure_sha: str) -> Path:
    """Publish the missing-section reason instead of fabricating comparisons."""
    target = int(STATS.CASES[case]["n_steps"])
    runs = report["legoesm_runs"]
    completed = [
        target,
        target,
        runs["fp64"]["first_nonfinite_completed_step"],
        runs["fp32"]["first_nonfinite_completed_step"],
    ]
    labels = ["NEMO\nFCT2", "NEMO\nFCT4", "legoESM\nfp64", "legoESM\nfp32"]
    colors = [ARM_COLORS[arm] for arm in ARMS]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.14, 1.0, 0.88))
    bars = ax.bar(labels, completed, color=colors)
    ax.axhline(target, color="black", linestyle="--", linewidth=1.0, label="required duration")
    for bar, value in zip(bars, completed):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02 * target,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, target * 1.13)
    ax.set_ylabel("Completed steps (legoESM stops at first non-finite state)")
    ax.set_title(
        f"{case}: full-duration comparison unavailable\n"
        "all preregistered metric verdicts = OUTSIDE",
        fontsize=13,
    )
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False)
    fields64 = ",".join(name for name, ok in runs["fp64"]["fields_finite"].items() if not ok)
    fields32 = ",".join(name for name, ok in runs["fp32"]["fields_finite"].items() if not ok)
    footer = (
        f"case={case}; required={target}; fp64 first non-finite={completed[2]} "
        f"({fields64}); fp32 first non-finite={completed[3]} ({fields32})\n"
        "No midpoint/final legoESM state, statistical metric, or three-time section "
        f"was invented; CPU artifacts loaded; figure_git={figure_sha}"
    )
    fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=7)
    path = output / f"{STATS.CASES[case]['slug']}_full_duration_failure.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Description": footer})
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    require(len(args.git_sha) == 40, "--git-sha must be a full commit")
    set_policy(PrecisionPolicy.fp64())
    require(bool(jax.config.jax_enable_x64), "plotting requires JAX x64")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.copy_dir.mkdir(parents=True, exist_ok=True)
    lego_root = args.full_root / "legoesm"
    report_root = args.full_root / "reports"
    reports = {}
    for case in STATS.CASES:
        path = report_root / f"{STATS.CASES[case]['slug']}.json"
        report = json.loads(path.read_text())
        require(report["case"] == case, f"{path}: case mismatch")
        reports[case] = report
    generated = []
    for case in ("OVERFLOW-zps", "LOCK_EXCHANGE-zco"):
        if reports[case]["metrics"]:
            generated.append(
                plot_sections(section_data(case, lego_root), args.output_dir, args.git_sha)
            )
            generated.append(plot_metrics(case, reports[case], args.output_dir, args.git_sha))
        else:
            generated.append(plot_failure(case, reports[case], args.output_dir, args.git_sha))
    copied = []
    for path in generated:
        destination = args.copy_dir / path.name
        shutil.copy2(path, destination)
        require(sha256(path) == sha256(destination), f"copy hash mismatch: {path}")
        copied.append(destination)
    print("FIGURES")
    for path in (*generated, *copied):
        print(f"{sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
