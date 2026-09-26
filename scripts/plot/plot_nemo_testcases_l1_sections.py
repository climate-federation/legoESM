#!/usr/bin/env python3
"""Plot the certified lane-1 NEMO testcase trajectory comparisons.

The section comparator deliberately imports the phase-3 gate's loaders and
frame adapters instead of reproducing them.  NEMO ``ts(...,Nbb)`` entry fields
are compared with legoESM's pre-step prognostic state after the gate's two-cell
halo strip on each horizontal side and its common wet-mask intersection.

No NEMO integration is launched.  legoESM is recomputed from each certified
card in CPU fp64 because trajectory states were not retained as artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_nemo_testcase_card,
)
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_PATH = (
    _REPO_ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_trajectory_gate.py"
)
_GATE_SPEC = importlib.util.spec_from_file_location("nemo_l1_phase3_gate", _GATE_PATH)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:
    raise ImportError(f"cannot load certified gate at {_GATE_PATH}")
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)
expected_masks = _GATE.expected_masks
lego_fields = _GATE.lego_fields
read_entry = _GATE.read_entry
require = _GATE.require

DEFAULT_ARTIFACT_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1")
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "figures"
DEFAULT_COPY = Path("/tmp/l1_figures")
PLOT_STEPS = (2, 10, 60)
CASES = {
    "LOCK_EXCHANGE-zco": {
        "slug": "lock_exchange_zco",
        "oracle": "phase3/lock_kt1_60",
        "gate": "phase3/lock_trajectory_gate_kt60.json",
        "title": "LOCK_EXCHANGE-zco",
    },
    "OVERFLOW-zps": {
        "slug": "overflow_zps",
        "oracle": "phase3/overflow_kt1_60",
        "gate": "phase3/overflow_trajectory_gate_kt60.json",
        "title": "OVERFLOW-zps",
    },
}
FRAME = (
    "T centres: NEMO ts(...,Nbb) entry vs legoESM pre-step prognostic T; "
    "two-cell halo stripped on each side; common wet-T mask; no interpolation"
)
ERROR_FRAME = (
    "gate normalized wet L-inf: max|candidate-oracle| / "
    "max(max|oracle|, 1); T centres and instantaneous prognostic U faces"
)


@dataclass(frozen=True)
class CaseData:
    case: str
    card: object
    row: int
    mask: np.ndarray
    h_partial: np.ndarray
    x_edges_km: np.ndarray
    bathymetry_m: np.ndarray
    oracle: dict[int, np.ndarray]
    candidate: dict[int, np.ndarray]


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_section_row(mask: np.ndarray) -> int:
    counts = np.count_nonzero(mask, axis=(1, 2))
    row = int(np.argmax(counts))
    require(counts[row] > 0, "section has no active T cells")
    require(
        np.count_nonzero(counts == counts[row]) == 1,
        f"section row is ambiguous: active-cell counts={counts.tolist()}",
    )
    return row


def _recompute_case(case: str, artifact_root: Path) -> CaseData:
    spec = CASES[case]
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
    )
    state = card.recipe.initial_state
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels

    print(
        f"{case}: backend={jax.default_backend()} policy={get_policy()} "
        f"T={state.T.data.dtype} u={state.u.data.dtype} "
        f"h_partial={card.recipe.z_coord.h_partial.dtype} "
        f"t_depth_ref={card.recipe.z_coord.t_depth_ref.dtype}"
    )
    for label, array in (
        ("T", state.T.data),
        ("u", state.u.data),
        ("h_partial", card.recipe.z_coord.h_partial),
        ("t_depth_ref", card.recipe.z_coord.t_depth_ref),
    ):
        require(np.asarray(array).dtype == np.float64, f"{case} {label} is not fp64")

    candidates: dict[int, np.ndarray] = {}
    for kt in range(1, max(PLOT_STEPS) + 1):
        if kt in PLOT_STEPS:
            candidates[kt] = np.array(lego_fields(state)["T"], copy=True)
        if kt < max(PLOT_STEPS):
            state = model.step(state, dt=card.dt_s)

    oracle_root = artifact_root / str(spec["oracle"])
    oracles: dict[int, np.ndarray] = {}
    for kt in PLOT_STEPS:
        entry_path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        entry = read_entry(entry_path, case)
        require(entry["step"] == kt, f"{entry_path}: step mismatch")
        oracles[kt] = np.asarray(entry["T"])[..., :nlev]
        require(
            oracles[kt].shape == candidates[kt].shape == masks["T"].shape,
            f"{case} kt={kt}: comparator shape mismatch",
        )
        active = masks["T"]
        require(
            np.all(np.isfinite(oracles[kt][active]))
            and np.all(np.isfinite(candidates[kt][active])),
            f"{case} kt={kt}: nonfinite wet T value",
        )

    row = _select_section_row(masks["T"])
    dx = np.asarray(card.recipe.grid.dx_T, dtype=np.float64)[row]
    x_edges_km = np.concatenate(([0.0], np.cumsum(dx))) / 1000.0
    return CaseData(
        case=case,
        card=card,
        row=row,
        mask=np.asarray(masks["T"], dtype=bool)[row],
        h_partial=np.asarray(card.recipe.z_coord.h_partial, dtype=np.float64)[row],
        x_edges_km=x_edges_km,
        bathymetry_m=np.asarray(card.recipe.initial_state.H_bathy.data, dtype=np.float64)[row],
        oracle={kt: values[row] for kt, values in oracles.items()},
        candidate={kt: values[row] for kt, values in candidates.items()},
    )


def _cell_polygons(data: CaseData) -> tuple[list[list[tuple[float, float]]], np.ndarray]:
    polygons: list[list[tuple[float, float]]] = []
    indices: list[tuple[int, int]] = []
    for ix in range(data.mask.shape[0]):
        interfaces = np.concatenate(([0.0], np.cumsum(data.h_partial[ix])))
        for iz in range(data.mask.shape[1]):
            if not data.mask[ix, iz]:
                continue
            x0, x1 = data.x_edges_km[ix : ix + 2]
            z0, z1 = interfaces[iz : iz + 2]
            require(z1 > z0, f"{data.case}: active cell has nonpositive thickness")
            polygons.append([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
            indices.append((ix, iz))
    require(bool(polygons), f"{data.case}: no section polygons")
    return polygons, np.asarray(indices, dtype=np.int64)


def _add_section(
    ax,
    data: CaseData,
    values: np.ndarray,
    polygons,
    indices: np.ndarray,
    *,
    cmap: str,
    norm: Normalize,
) -> PolyCollection:
    plotted = values[indices[:, 0], indices[:, 1]]
    require(np.all(np.isfinite(plotted)), f"{data.case}: nonfinite plotted values")
    collection = PolyCollection(
        polygons,
        array=plotted,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=True,
    )
    ax.add_collection(collection)
    depth_max = float(np.max(data.bathymetry_m))
    bottom = np.r_[data.bathymetry_m, data.bathymetry_m[-1]]
    ax.fill_between(
        data.x_edges_km,
        bottom,
        depth_max * 1.025,
        step="post",
        color="0.78",
        zorder=3,
    )
    ax.step(
        data.x_edges_km,
        bottom,
        where="post",
        color="black",
        linewidth=1.0,
        zorder=4,
    )
    ax.set_xlim(data.x_edges_km[0], data.x_edges_km[-1])
    ax.set_ylim(depth_max * 1.025, 0.0)
    ax.grid(False)
    return collection


def _scientific_colorbar(colorbar) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    colorbar.formatter = formatter
    colorbar.update_ticks()


def _plot_case(data: CaseData, output: Path, git_sha: str) -> Path:
    polygons, indices = _cell_polygons(data)
    wet_values = []
    for kt in PLOT_STEPS:
        wet_values.extend(data.oracle[kt][data.mask])
        wet_values.extend(data.candidate[kt][data.mask])
    state_norm = Normalize(float(np.min(wet_values)), float(np.max(wet_values)))

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.2, 8.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(rect=(0.0, 0.075, 1.0, 0.875))
    fig.suptitle(f"{CASES[data.case]['title']}: temperature sections", y=0.985)
    state_mappable = None
    diff_limits: list[float] = []
    for column, kt in enumerate(PLOT_STEPS):
        oracle = data.oracle[kt]
        candidate = data.candidate[kt]
        difference = candidate - oracle
        vmax = float(np.max(np.abs(difference[data.mask])))
        require(vmax > 0.0, f"{data.case} kt={kt}: vacuous zero difference")
        diff_limits.append(vmax)
        oracle_scale = max(float(np.max(np.abs(oracle[data.mask]))), 1.0)
        normalized_vmax = vmax / oracle_scale
        diff_norm = Normalize(-vmax, vmax)

        state_mappable = _add_section(
            axes[0, column],
            data,
            oracle,
            polygons,
            indices,
            cmap="inferno",
            norm=state_norm,
        )
        _add_section(
            axes[1, column],
            data,
            candidate,
            polygons,
            indices,
            cmap="inferno",
            norm=state_norm,
        )
        diff_mappable = _add_section(
            axes[2, column],
            data,
            difference,
            polygons,
            indices,
            cmap="RdBu_r",
            norm=diff_norm,
        )
        axes[0, column].set_title(f"kt = {kt}")
        axes[2, column].set_title(
            rf"legoESM - NEMO; scale $\pm${vmax:.2e} K"
            f"; norm. {normalized_vmax:.2e}",
            fontsize=9.0,
        )
        colorbar = fig.colorbar(
            diff_mappable,
            ax=axes[2, column],
            orientation="horizontal",
            pad=0.15,
            fraction=0.08,
        )
        _scientific_colorbar(colorbar)
        colorbar.set_label("Temperature difference (K)")

    row_labels = ("NEMO oracle", "legoESM", "Difference")
    for row_index, label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(f"{label}\nDepth (m)")
    for ax in axes[-1]:
        ax.set_xlabel("x (km)")
    require(state_mappable is not None, "missing state mappable")
    state_colorbar = fig.colorbar(
        state_mappable,
        ax=axes[:2, :],
        location="right",
        pad=0.015,
        fraction=0.025,
    )
    state_colorbar.set_label("Temperature (°C)")

    footer = (
        f"case={data.case}; kt={','.join(map(str, PLOT_STEPS))}; frame={FRAME}\n"
        f"geometry=certified h_partial; legoESM=recomputed CPU fp64; git={git_sha}"
    )
    fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=6.8)
    path = output / f"{CASES[data.case]['slug']}_temperature_sections.png"
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": f"{data.case} lane-1 temperature sections",
            "Description": footer,
            "Software": f"legoESM {git_sha}",
        },
    )
    plt.close(fig)
    scale_rows = []
    for kt, limit in zip(PLOT_STEPS, diff_limits):
        oracle_scale = max(float(np.max(np.abs(data.oracle[kt][data.mask]))), 1.0)
        scale_rows.append(f"kt{kt}:physical={limit:.17e}K,normalized={limit / oracle_scale:.17e}")
    print(f"{data.case}: diff scales " + ", ".join(scale_rows))
    return path


def _gate_series(report: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    errors = []
    for step in report["steps"]:
        row = next(row for row in step["rows"] if row["name"].endswith(f".{field}"))
        error = float(row["normalized_max_abs"])
        if error > 0.0:
            steps.append(int(step["kt"]))
            errors.append(error)
    require(bool(steps), f"{report['case']} {field}: no positive gate errors")
    return np.asarray(steps), np.asarray(errors)


def _plot_summary(artifact_root: Path, output: Path, git_sha: str) -> Path:
    reports = {}
    gate_hashes = {}
    for case, spec in CASES.items():
        gate_path = artifact_root / str(spec["gate"])
        report = json.loads(gate_path.read_text())
        require(report["case"] == case, f"{gate_path}: case mismatch")
        require(len(report["steps"]) == 60, f"{gate_path}: expected 60 steps")
        reports[case] = report
        gate_hashes[case] = _sha256(gate_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.095, 1.0, 0.855))
    colors = {"LOCK_EXCHANGE-zco": "#2166ac", "OVERFLOW-zps": "#b2182b"}
    for ax, field, title in zip(
        axes,
        ("T", "u"),
        ("T-centre error", "instantaneous U-face error"),
    ):
        for case in CASES:
            step, error = _gate_series(reports[case], field)
            growth = reports[case]["growth_characterization"][field]
            require(growth["status"] == "MEASURED", f"{case} {field}: no exponent")
            exponent = float(growth["tail_power_law_exponent_p"])
            ax.plot(
                step,
                error,
                color=colors[case],
                linewidth=1.8,
                marker="o",
                markersize=2.4,
                markevery=5,
                label=f"{CASES[case]['title']}  (tail p={exponent:.2f})",
            )
        ax.set_yscale("log")
        ax.set_xlabel("NEMO step-entry kt")
        ax.set_ylabel(r"Normalized wet $L_\infty$ error")
        ax.set_title(title)
        ax.grid(True, which="both", color="0.87", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Lane 1 trajectory-error growth through kt=60", y=0.985)
    footer = (
        f"cases=LOCK_EXCHANGE-zco,OVERFLOW-zps; kt=1..60; frame={ERROR_FRAME}\n"
        f"tail p read from certified kt60 gate JSON; git={git_sha}; "
        f"gate_sha256(lock)={gate_hashes['LOCK_EXCHANGE-zco'][:12]}; "
        f"gate_sha256(overflow)={gate_hashes['OVERFLOW-zps'][:12]}"
    )
    fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=6.8)
    path = output / "lane1_temperature_velocity_error_growth.png"
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": "Lane-1 certified trajectory-error growth",
            "Description": footer,
            "Software": f"legoESM {git_sha}",
        },
    )
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    parser.add_argument(
        "--git-sha",
        help="commit to stamp; defaults to the current checkout's HEAD",
    )
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "plot recomputation must use CPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.copy_dir.mkdir(parents=True, exist_ok=True)
    git_sha = args.git_sha or _git_sha()
    require(len(git_sha) == 40, f"git SHA is not a full 40-character hash: {git_sha}")

    cases = {case: _recompute_case(case, args.artifact_root) for case in CASES}
    generated = [
        _plot_case(cases["OVERFLOW-zps"], args.output_dir, git_sha),
        _plot_case(cases["LOCK_EXCHANGE-zco"], args.output_dir, git_sha),
        _plot_summary(args.artifact_root, args.output_dir, git_sha),
    ]
    copied = []
    for path in generated:
        destination = args.copy_dir / path.name
        shutil.copy2(path, destination)
        require(_sha256(path) == _sha256(destination), f"copy mismatch: {path}")
        copied.append(destination)

    print("FIGURES")
    for path in (*generated, *copied):
        print(f"{_sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
