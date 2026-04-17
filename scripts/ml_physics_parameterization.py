#!/usr/bin/env python
"""Run the analytical AMIP baseline, train the joint ML parameterization, and compare."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig, OutputConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ml.physics import (
    DEFAULT_SAMPLE_DAYS,
    PhysicsModelConfig,
    PhysicsTrainingConfig,
    capture_physics_teacher_snapshot,
    concatenate_physics_teacher_datasets,
    predict_physics_parameterization_targets,
    save_physics_checkpoint,
    save_physics_stats,
    train_physics_parameterization,
)
from legoesm.ml.physics.plotting import (
    plot_profile_rmse,
    plot_rollout_difference_maps,
    plot_rollout_map_fields,
    plot_rollout_profiles,
    plot_rollout_timeseries,
    plot_sample_profile_comparison,
    plot_training_history,
)


DEFAULT_LABEL = "Default"
FULL_ML_LABEL = "Full ML"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI for the analytical ML physics workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/ml_physics_parameterization"),
    )
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--diag-days", type=int, default=1)
    parser.add_argument(
        "--sample-days",
        type=str,
        default=",".join(str(int(day)) for day in DEFAULT_SAMPLE_DAYS),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--microphysics",
        choices=("none", "kessler", "sundqvist"),
        default="none",
    )
    parser.add_argument("--rad-update-steps", type=int, default=3)
    return parser


def _sample_days_from_args(args: argparse.Namespace) -> tuple[float, ...]:
    """Parse the comma-separated sample-day list from the CLI."""
    return tuple(float(chunk.strip()) for chunk in args.sample_days.split(",") if chunk.strip())


def _generate_standard_run_plots(run_dir: Path) -> None:
    """Regenerate the standard AMIP diagnostic plots for one run directory."""
    plot_amip_path = REPO_ROOT / "scripts" / "diagnostic" / "plot_amip.py"
    spec = importlib.util.spec_from_file_location("plot_amip_module", plot_amip_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load plot_amip from {plot_amip_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.plot_amip(run_dir, show=False)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load a NumPy ``.npz`` bundle into a plain mapping."""
    with np.load(path) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _is_moist_microphysics(microphysics: str) -> bool:
    """Return whether the selected workflow path uses moist physics."""
    return microphysics != "none"


def _resolve_cloud_scheme(microphysics: str) -> str:
    """Choose the cloud-fraction scheme paired with the selected microphysics."""
    if microphysics == "kessler":
        return "xu_randall"
    if microphysics == "sundqvist":
        return "sundqvist"
    return "none"


def _extract_final_snapshot_fields(
    snapshots: dict[str, np.ndarray],
) -> tuple[str, dict[str, np.ndarray]]:
    """Extract the last saved map fields used in the comparison plots."""
    final_day = int(np.asarray(snapshots["snapshot_days"], dtype=int)[-1])
    day_label = f"{final_day:03d}"
    fields = {
        "T_low": snapshots[f"day{day_label}_T_low"],
        "q_v_low": snapshots[f"day{day_label}_q_v_low"],
        "precip": snapshots[f"day{day_label}_precip"],
        "wind": snapshots[f"day{day_label}_wind"],
    }
    for optional_name in ("q_c_low", "q_r_low"):
        key = f"day{day_label}_{optional_name}"
        if key in snapshots:
            fields[optional_name] = snapshots[key]
    return day_label, fields


def _make_base_config(args: argparse.Namespace, output_dir: Path) -> ExperimentConfig:
    """Build the physical configuration shared by the workflow stages."""
    use_moist_microphysics = _is_moist_microphysics(args.microphysics)
    return ExperimentConfig(
        grid=GridConfig(
            grid_type="cubed_sphere",
            resolution=args.resolution,
            nlev=args.nlev,
            vertical_coord="hybrid",
            p_top_Pa=200.0,
            stretching=2.0,
        ),
        dycore=DycoreConfig(dt=args.dt),
        output=OutputConfig(
            output_dir=str(output_dir),
            diag_days=args.diag_days,
            checkpoint_days=0,
        ),
        days=args.days,
        dataset="analytical",
        radiation="rrtmgp" if use_moist_microphysics else "gray",
        rad_update_steps=args.rad_update_steps if use_moist_microphysics else 1,
        ozone_source="analytical" if use_moist_microphysics else "standard",
        cloud_scheme=_resolve_cloud_scheme(args.microphysics),
        microphysics=args.microphysics,
        convection="mass_flux",
        turbulence="louis",
    )


def _run_default_and_capture_dataset(
    args: argparse.Namespace,
    output_dir: Path,
) -> object:
    """Run the physical baseline and capture teacher columns on sampled days."""
    config = _make_base_config(args, output_dir)
    sample_days = _sample_days_from_args(args)
    driver = ModelDriver(config, output_dir=output_dir)
    driver.setup()

    dt = float(config.dycore.dt)
    pending_days = list(sample_days)
    captured: dict[float, object] = {}

    def _capture_matching_days(current_day: float) -> None:
        tol = max(dt / 86400.0, 1e-8)
        ready = [day for day in pending_days if current_day + tol >= day]
        for day in ready:
            captured[day] = capture_physics_teacher_snapshot(driver, day, dt)
            pending_days.remove(day)

    _capture_matching_days(config.start_day)

    def _segment_callback(driver_obj, day, dt_segment):
        del driver_obj, dt_segment
        _capture_matching_days(float(day))

    status = driver.run(segment_callback=_segment_callback)
    if status != "COMPLETED":
        raise RuntimeError(f"default analytical AMIP run failed: {status}")
    if pending_days:
        raise RuntimeError(f"did not capture requested sample days: {pending_days}")

    _generate_standard_run_plots(output_dir)
    ordered = [captured[day] for day in sample_days]
    dataset = concatenate_physics_teacher_datasets(ordered)
    expected = len(sample_days) * 6 * args.resolution * args.resolution
    if dataset.columns.T.shape[0] != expected:
        raise RuntimeError(
            f"expected {expected} sampled columns, got {dataset.columns.T.shape[0]}",
        )
    return dataset


def _train_joint_model(args: argparse.Namespace, dataset, training_dir: Path) -> tuple[Path, Path, object]:
    """Train the joint model and write its checkpoint, stats, and diagnostics."""
    training_dir.mkdir(parents=True, exist_ok=True)
    result = train_physics_parameterization(
        dataset,
        model_config=PhysicsModelConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.layers,
            seed=args.seed,
        ),
        training_config=PhysicsTrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        ),
    )
    checkpoint_path = training_dir / "physics_parameterization.eqx"
    stats_path = training_dir / "physics_parameterization_stats.npz"
    save_physics_checkpoint(result.model, checkpoint_path)
    save_physics_stats(result.stats_bundle, stats_path)

    predictions = predict_physics_parameterization_targets(
        model=result.model,
        stats_bundle=result.stats_bundle,
        columns=dataset.columns,
    )
    sample_day = np.asarray(dataset.columns.day, dtype=float)
    day1_matches = np.where(np.isclose(sample_day, 1.0))[0]
    sample_index = int(day1_matches[0]) if day1_matches.size else 0
    teacher = {
        "Km": np.asarray(dataset.Km),
        "Kh": np.asarray(dataset.Kh),
        "M_eq": np.asarray(dataset.M_eq),
    }
    student = {
        "Km": np.asarray(predictions["Km"]),
        "Kh": np.asarray(predictions["Kh"]),
        "M_eq": np.asarray(predictions["M_eq"]),
    }
    plot_training_history(
        result.train_loss_history,
        result.val_loss_history,
        training_dir / "training_loss.png",
    )
    plot_sample_profile_comparison(
        np.asarray(dataset.columns.p_full),
        teacher,
        student,
        training_dir / "sample_profile_comparison.png",
        sample_index=sample_index,
        sample_day=float(sample_day[sample_index]),
    )
    plot_profile_rmse(
        np.asarray(dataset.columns.p_full),
        teacher,
        student,
        training_dir / "profile_rmse.png",
    )
    metrics = {
        "best_train_loss": result.best_train_loss,
        "best_val_loss": result.best_val_loss,
        "test_loss": result.test_loss,
        "epochs_ran": result.epochs_ran,
        "best_epoch": result.best_epoch,
        "rmse_Km": result.metrics.rmse_Km,
        "rmse_Kh": result.metrics.rmse_Kh,
        "rmse_M_eq": result.metrics.rmse_M_eq,
        "rmse_rain_survival_fraction": result.metrics.rmse_rain_survival_fraction,
        "rmse_dq_v_dt_micro": result.metrics.rmse_dq_v_dt_micro,
        "rmse_dq_c_dt_micro": result.metrics.rmse_dq_c_dt_micro,
        "rmse_dq_r_dt_micro": result.metrics.rmse_dq_r_dt_micro,
        "rmse_precip_micro": result.metrics.rmse_precip_micro,
        "microphysics_scheme": dataset.microphysics_scheme,
        "n_columns": int(dataset.columns.T.shape[0]),
        "sample_days": list(dataset.sample_days),
    }
    (training_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return checkpoint_path, stats_path, metrics


def _run_full_ml(
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: Path,
    stats_path: Path,
) -> None:
    """Run the online-coupled ML rollout using saved model assets."""
    config = _make_base_config(args, output_dir)._replace(
        physics_parameterization="ml",
        physics_parameterization_checkpoint=str(checkpoint_path),
        physics_parameterization_stats=str(stats_path),
        physics_parameterization_hidden_dim=args.hidden_dim,
        physics_parameterization_layers=args.layers,
        physics_parameterization_seed=args.seed,
    )
    driver = ModelDriver(config, output_dir=output_dir)
    driver.setup()
    status = driver.run()
    if status != "COMPLETED":
        raise RuntimeError(f"full-ml analytical AMIP run failed: {status}")
    _generate_standard_run_plots(output_dir)


def _build_comparison(output_root: Path) -> dict[str, float]:
    """Build the baseline-vs-ML comparison package for a completed workflow."""
    compare_dir = output_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    default_dir = output_root / "default_run"
    full_ml_dir = output_root / "full_ml_run"

    default_ts = _load_npz(default_dir / "timeseries.npz")
    full_ml_ts = _load_npz(full_ml_dir / "timeseries.npz")
    default_snapshots = _load_npz(default_dir / "snapshots.npz")
    full_ml_snapshots = _load_npz(full_ml_dir / "snapshots.npz")
    day_label, default_fields = _extract_final_snapshot_fields(default_snapshots)
    full_ml_day_label, full_ml_fields = _extract_final_snapshot_fields(full_ml_snapshots)
    if full_ml_day_label != day_label:
        raise RuntimeError(
            f"snapshot day mismatch between default ({day_label}) and full_ml ({full_ml_day_label})",
        )

    plot_rollout_timeseries(
        default_ts,
        full_ml_ts,
        compare_dir / "timeseries_compare.png",
        baseline_label=DEFAULT_LABEL,
        ml_label=FULL_ML_LABEL,
    )
    plot_rollout_profiles(
        default_ts,
        full_ml_ts,
        compare_dir / "final_profiles_compare.png",
        baseline_label=DEFAULT_LABEL,
        ml_label=FULL_ML_LABEL,
    )

    grid = create_cubed_sphere(int(np.asarray(default_fields["T_low"]).shape[1]))
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    plot_rollout_map_fields(
        lon_deg,
        lat_deg,
        default_fields,
        full_ml_fields,
        compare_dir / "maps_final.png",
        day_label=day_label,
        baseline_label=DEFAULT_LABEL,
        ml_label=FULL_ML_LABEL,
    )
    plot_rollout_difference_maps(
        lon_deg,
        lat_deg,
        default_fields,
        full_ml_fields,
        compare_dir / "difference_maps_final.png",
        day_label=day_label,
        baseline_label=DEFAULT_LABEL,
        ml_label=FULL_ML_LABEL,
    )

    summary: dict[str, float] = {}
    summary_keys = (
        "T_atm",
        "T_low",
        "CWV",
        "precip",
        "max_wind",
        "sw_up_toa",
        "lw_up_toa",
        "sw_net_sfc",
        "lw_net_sfc",
    )
    for key in summary_keys:
        if key not in default_ts or key not in full_ml_ts:
            continue
        summary[f"final_{key}_default"] = float(default_ts[key][-1])
        summary[f"final_{key}_full_ml"] = float(full_ml_ts[key][-1])
        summary[f"delta_{key}_full_ml_minus_default"] = float(
            full_ml_ts[key][-1] - default_ts[key][-1]
        )
    (compare_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_workflow(args: argparse.Namespace) -> int:
    """Execute the full baseline-train-rollout comparison workflow."""
    output_root = args.output_root
    training_dir = output_root / "training"
    default_run_dir = output_root / "default_run"
    full_ml_run_dir = output_root / "full_ml_run"
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = _run_default_and_capture_dataset(args, default_run_dir)
    checkpoint_path, stats_path, training_metrics = _train_joint_model(
        args,
        dataset,
        training_dir,
    )
    _run_full_ml(args, full_ml_run_dir, checkpoint_path, stats_path)
    comparison_summary = _build_comparison(output_root)
    workflow_summary = {
        "training_dir": str(training_dir),
        "default_run_dir": str(default_run_dir),
        "full_ml_run_dir": str(full_ml_run_dir),
        "comparison_dir": str(output_root / "comparison"),
        "training_metrics": training_metrics,
        "comparison_summary": comparison_summary,
    }
    (output_root / "workflow_summary.json").write_text(
        json.dumps(workflow_summary, indent=2) + "\n",
    )
    print(f"completed ML physics parameterization workflow under {output_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the analytical ML physics workflow."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
