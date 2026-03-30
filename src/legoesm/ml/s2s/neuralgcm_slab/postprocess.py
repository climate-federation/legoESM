"""Postprocessing and plotting for NeuralGCM slab-ocean ensemble cases."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from legoesm.ml.s2s.plotting import frame_to_image as _frame_to_image, plot_latlon_map as _plot_map
from legoesm.ml.s2s.neuralgcm_slab.ensemble import (
    HEADLINE_FIELDS,
    HeadlineField,
    aggregate_daily_series,
    build_ensemble_daily_crps_table,
    build_ensemble_mean_daily_mae_table,
    build_ensemble_mean_daily_rmse_table,
    build_member_daily_mae_table,
    build_member_daily_rmse_table,
    open_member_forecasts,
    save_long_records_csv,
)
from legoesm.ml.s2s.neuralgcm_slab.evaluation import (
    DEFAULT_S2S_WINDOWS,
    LeadTimeWindow,
    aggregate_long_records,
    load_long_record_csv,
)
from legoesm.ml.s2s.neuralgcm_slab.metrics import align_truth_to_forecast
from legoesm.ml.s2s.sfno_slab.postprocess import (
    postprocess_campaign_center_crps as sfno_postprocess_campaign_center_crps,
)


EXPERIMENTS = ("coupled", "uncoupled")
FIELD_LABELS = {
    "t850": "t-850",
    "z500": "z-500",
    "q700": "q-700",
    "sst": "sosstsst",
}
FIELD_STEMS = {value: key for key, value in FIELD_LABELS.items()}
FIELD_TITLES = {
    "t-850": "T850",
    "z-500": "Z500",
    "q-700": "Q700",
    "sosstsst": "SST",
}
BOX_COLORS = {"coupled": "#1f77b4", "uncoupled": "#ff7f0e"}


def _open_dataset(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _window_sequence(windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]]) -> tuple[LeadTimeWindow, ...]:
    if isinstance(windows, Mapping):
        return tuple(
            LeadTimeWindow(name, start_day, end_day)
            for name, (start_day, end_day) in windows.items()
        )
    return tuple(windows)


def _window_map(windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    sequence = _window_sequence(windows)
    return {window.name: (window.start_day, window.end_day) for window in sequence}


def _headline_lookup(fields: Sequence[HeadlineField] = HEADLINE_FIELDS) -> dict[str, HeadlineField]:
    return {field.name: field for field in fields}


def _headline_from_label(label: str, fields: Sequence[HeadlineField] = HEADLINE_FIELDS) -> HeadlineField:
    lookup = {FIELD_LABELS[field.name]: field for field in fields}
    return lookup[label]


def _field_label(field: HeadlineField) -> str:
    return FIELD_LABELS[field.name]


def _select_plot_field(dataset: xr.Dataset, field: HeadlineField) -> xr.DataArray:
    data = dataset[field.variable]
    if field.level is not None:
        data = data.sel(level=field.level, method="nearest")
    return data * field.scale


def _lead_days(dataset: xr.Dataset) -> np.ndarray:
    if "lead_day" in dataset.coords:
        return np.asarray(dataset["lead_day"], dtype=int)
    return np.arange(1, int(dataset.sizes.get("lead_day", 0)) + 1, dtype=int)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return save_long_records_csv(rows, path)


def _prepared_dataset_path(case_dir: Path, filename: str) -> Path:
    return case_dir / "_prepared" / filename


def _case_metric_path(case_dir: Path, filename: str) -> Path:
    return case_dir / "metrics" / filename


def _save_metric_boxplots(
    ensemble_daily_rows: Sequence[Mapping[str, Any]],
    *,
    windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]],
    output_dir: Path,
) -> None:
    window_map = _window_map(windows)
    output_dir.mkdir(parents=True, exist_ok=True)
    for field_label in FIELD_STEMS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        for ax, metric in zip(axes, ("rmse", "crps"), strict=True):
            positions: list[float] = []
            data: list[np.ndarray] = []
            facecolors: list[str] = []
            for window_index, (window_name, (start_day, end_day)) in enumerate(window_map.items()):
                center = float(window_index)
                for experiment, offset in (("coupled", -0.18), ("uncoupled", 0.18)):
                    values = [
                        float(record[metric])
                        for record in ensemble_daily_rows
                        if record["field"] == field_label
                        and record["experiment"] == experiment
                        and start_day <= int(record["lead_day"]) <= end_day
                    ]
                    if not values:
                        continue
                    positions.append(center + offset)
                    data.append(np.asarray(values, dtype=float))
                    facecolors.append(BOX_COLORS[experiment])
            if data:
                artists = ax.boxplot(
                    data,
                    positions=positions,
                    widths=0.28,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, color in zip(artists["boxes"], facecolors, strict=True):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                for median in artists["medians"]:
                    median.set_color("black")
            ax.set_xticks(range(len(window_map)), labels=list(window_map))
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{FIELD_TITLES[field_label]} {metric.upper()}")
            ax.grid(True, alpha=0.3)
            handles = [
                plt.Line2D([0], [0], color=BOX_COLORS["coupled"], lw=8, label="coupled"),
                plt.Line2D([0], [0], color=BOX_COLORS["uncoupled"], lw=8, label="uncoupled"),
            ]
            ax.legend(handles=handles)
        fig.savefig(output_dir / f"{FIELD_STEMS[field_label]}_metric_summary.png", dpi=150)
        plt.close(fig)


def _save_campaign_metric_figures(
    case_window_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]],
) -> None:
    window_names = [window.name for window in _window_sequence(windows)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for field_label in FIELD_STEMS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        for ax, metric in zip(axes, ("rmse", "crps"), strict=True):
            positions: list[float] = []
            data: list[np.ndarray] = []
            facecolors: list[str] = []
            for window_index, window_name in enumerate(window_names):
                center = float(window_index)
                for experiment, offset in (("coupled", -0.18), ("uncoupled", 0.18)):
                    values = [
                        float(record["value"])
                        for record in case_window_records
                        if record["field"] == field_label
                        and record["metric"] == metric
                        and record["window"] == window_name
                        and record["experiment"] == experiment
                    ]
                    if not values:
                        continue
                    positions.append(center + offset)
                    data.append(np.asarray(values, dtype=float))
                    facecolors.append(BOX_COLORS[experiment])
            if data:
                artists = ax.boxplot(
                    data,
                    positions=positions,
                    widths=0.28,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, color in zip(artists["boxes"], facecolors, strict=True):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                for median in artists["medians"]:
                    median.set_color("black")
            ax.set_xticks(range(len(window_names)), labels=window_names)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{FIELD_TITLES[field_label]} {metric.upper()}")
            ax.grid(True, alpha=0.3)
            handles = [
                plt.Line2D([0], [0], color=BOX_COLORS["coupled"], lw=8, label="coupled"),
                plt.Line2D([0], [0], color=BOX_COLORS["uncoupled"], lw=8, label="uncoupled"),
            ]
            ax.legend(handles=handles)
        fig.savefig(output_dir / f"{FIELD_STEMS[field_label]}_campaign_metric_summary.png", dpi=150)
        plt.close(fig)


def _save_member_snapshot_figures(
    coupled_members: Sequence[xr.Dataset],
    uncoupled_members: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    field: HeadlineField,
    snapshot_days: Sequence[int],
    plot_member_indices: Sequence[int],
    output_dir: Path,
) -> None:
    reference = coupled_members[0]
    aligned_truth = align_truth_to_forecast(reference, truth)
    lead_to_index = {int(day): index for index, day in enumerate(_lead_days(reference).tolist())}
    truth_fields = [
        np.asarray(_select_plot_field(aligned_truth, field).isel(lead_day=lead_to_index[day]), dtype=float)
        for day in snapshot_days
        if day in lead_to_index
    ]
    if not truth_fields:
        return
    vmin = float(np.nanmin(truth_fields))
    vmax = float(np.nanmax(truth_fields))

    for member_index in plot_member_indices:
        coupled = coupled_members[member_index]
        uncoupled = uncoupled_members[member_index]
        fig, axes = plt.subplots(len(snapshot_days), 3, figsize=(18, 4 * len(snapshot_days)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row_index, day in enumerate(snapshot_days):
            if day not in lead_to_index:
                continue
            lead_index = lead_to_index[day]
            truth_field = _select_plot_field(aligned_truth, field).isel(lead_day=lead_index)
            coupled_field = _select_plot_field(coupled, field).isel(lead_day=lead_index)
            uncoupled_field = _select_plot_field(uncoupled, field).isel(lead_day=lead_index)
            _plot_map(axes[row_index, 0], truth_field, f"Truth {field.title}, day {day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[row_index, 1], coupled_field, f"Coupled {field.title}, day {day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[row_index, 2], uncoupled_field, f"Uncoupled {field.title}, day {day}", vmin=vmin, vmax=vmax)
        fig.suptitle(f"Member {member_index:02d}", fontsize=14)
        fig.colorbar(axes[-1, 2].collections[0], ax=axes, shrink=0.96, label=field.colorbar_label)
        fig.savefig(output_dir / f"{field.name}_member{member_index:02d}_snapshots.png", dpi=150)
        plt.close(fig)


def _save_member_gifs(
    coupled_members: Sequence[xr.Dataset],
    uncoupled_members: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    field: HeadlineField,
    plot_member_indices: Sequence[int],
    output_dir: Path,
    duration: float = 0.45,
) -> None:
    if imageio is None:
        raise ImportError("imageio is required for GIF generation")
    reference = coupled_members[0]
    aligned_truth = align_truth_to_forecast(reference, truth)
    lead_days = _lead_days(reference).tolist()
    truth_values = [
        np.asarray(_select_plot_field(aligned_truth, field).isel(lead_day=index), dtype=float)
        for index in range(len(lead_days))
    ]
    vmin = float(np.nanmin(truth_values))
    vmax = float(np.nanmax(truth_values))

    for member_index in plot_member_indices:
        coupled = coupled_members[member_index]
        uncoupled = uncoupled_members[member_index]
        frames: list[np.ndarray] = []
        for lead_index, lead_day in enumerate(lead_days):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            truth_field = _select_plot_field(aligned_truth, field).isel(lead_day=lead_index)
            coupled_field = _select_plot_field(coupled, field).isel(lead_day=lead_index)
            uncoupled_field = _select_plot_field(uncoupled, field).isel(lead_day=lead_index)
            _plot_map(axes[0], truth_field, f"Truth {field.title}, day {lead_day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[1], coupled_field, f"Coupled {field.title}, day {lead_day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[2], uncoupled_field, f"Uncoupled {field.title}, day {lead_day}", vmin=vmin, vmax=vmax)
            fig.colorbar(axes[2].collections[0], ax=axes, shrink=0.9, label=field.colorbar_label)
            fig.suptitle(f"Member {member_index:02d}", fontsize=12)
            frames.append(_frame_to_image(fig))
        imageio.mimsave(
            output_dir / f"{field.name}_member{member_index:02d}_daily.gif",
            frames,
            duration=duration,
            loop=0,
        )


def _member_daily_rows(
    forecasts: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    experiment: str,
    fields: Sequence[HeadlineField],
    windows: Sequence[LeadTimeWindow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    daily_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for member_index, forecast in enumerate(forecasts):
        rmse = build_member_daily_rmse_table(forecast, truth, fields=fields)
        mae = build_member_daily_mae_table(forecast, truth, fields=fields)
        lead_days = _lead_days(forecast)
        rmse_windows = aggregate_daily_series(rmse, lead_days=lead_days, windows=windows)
        mae_windows = aggregate_daily_series(mae, lead_days=lead_days, windows=windows)
        for field in fields:
            field_label = _field_label(field)
            for lead_day, rmse_value, mae_value in zip(
                lead_days,
                rmse[field.name],
                mae[field.name],
                strict=True,
            ):
                daily_rows.append(
                    {
                        "experiment": experiment,
                        "member": member_index,
                        "field": field_label,
                        "lead_day": int(lead_day),
                        "rmse": float(rmse_value),
                        "mae": float(mae_value),
                    }
                )
            for window in windows:
                window_rows.append(
                    {
                        "experiment": experiment,
                        "member": member_index,
                        "field": field_label,
                        "window": window.name,
                        "rmse": float(rmse_windows[window.name][field.name]),
                        "mae": float(mae_windows[window.name][field.name]),
                    }
                )
    return daily_rows, window_rows


def _ensemble_daily_rows(
    forecasts: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    experiment: str,
    fields: Sequence[HeadlineField],
    windows: Sequence[LeadTimeWindow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = forecasts[0]
    lead_days = _lead_days(reference)
    rmse = build_ensemble_mean_daily_rmse_table(forecasts, truth, fields=fields)
    mae = build_ensemble_mean_daily_mae_table(forecasts, truth, fields=fields)
    crps = build_ensemble_daily_crps_table(forecasts, truth, fields=fields)
    rmse_windows = aggregate_daily_series(rmse, lead_days=lead_days, windows=windows)
    mae_windows = aggregate_daily_series(mae, lead_days=lead_days, windows=windows)
    crps_windows = aggregate_daily_series(crps, lead_days=lead_days, windows=windows)

    daily_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for field in fields:
        field_label = _field_label(field)
        for lead_day, rmse_value, mae_value, crps_value in zip(
            lead_days,
            rmse[field.name],
            mae[field.name],
            crps[field.name],
            strict=True,
        ):
            daily_rows.append(
                {
                    "experiment": experiment,
                    "field": field_label,
                    "lead_day": int(lead_day),
                    "rmse": float(rmse_value),
                    "mae": float(mae_value),
                    "crps": float(crps_value),
                    "n_members": int(len(forecasts)),
                }
            )
        for window in windows:
            window_rows.append(
                {
                    "experiment": experiment,
                    "field": field_label,
                    "window": window.name,
                    "rmse": float(rmse_windows[window.name][field.name]),
                    "mae": float(mae_windows[window.name][field.name]),
                    "crps": float(crps_windows[window.name][field.name]),
                    "n_members": int(len(forecasts)),
                }
            )
    return daily_rows, window_rows


def postprocess_case(
    case_dir: str | Path,
    *,
    n_members: int,
    plot_member_indices: Sequence[int] = (0, 1),
    windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]] = DEFAULT_S2S_WINDOWS,
    fields: Sequence[HeadlineField] = HEADLINE_FIELDS,
    snapshot_days: Sequence[int] = (1, 15, 29, 42),
    make_snapshots: bool = True,
    make_gifs: bool = True,
) -> dict[str, Path]:
    case_dir = Path(case_dir)
    metrics_dir = case_dir / "metrics"
    plots_dir = case_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    truth = _open_dataset(_prepared_dataset_path(case_dir, "truth.nc"))
    member_indices = list(range(int(n_members)))
    plot_members = [index for index in plot_member_indices if 0 <= index < len(member_indices)]
    windows_seq = _window_sequence(windows)

    member_daily_rows: list[dict[str, Any]] = []
    member_window_rows: list[dict[str, Any]] = []
    ensemble_daily_rows: list[dict[str, Any]] = []
    ensemble_window_rows: list[dict[str, Any]] = []
    forecasts_by_experiment: dict[str, list[xr.Dataset]] = {}

    for experiment in EXPERIMENTS:
        forecasts = open_member_forecasts(
            case_dir,
            member_indices=member_indices,
            experiment=experiment,
        )
        forecasts_by_experiment[experiment] = forecasts
        member_daily, member_window = _member_daily_rows(
            forecasts,
            truth,
            experiment=experiment,
            fields=fields,
            windows=windows_seq,
        )
        ensemble_daily, ensemble_window = _ensemble_daily_rows(
            forecasts,
            truth,
            experiment=experiment,
            fields=fields,
            windows=windows_seq,
        )
        member_daily_rows.extend(member_daily)
        member_window_rows.extend(member_window)
        ensemble_daily_rows.extend(ensemble_daily)
        ensemble_window_rows.extend(ensemble_window)

    outputs = {
        "daily_ensemble_metrics": _write_rows(metrics_dir / "daily_ensemble_metrics.csv", ensemble_daily_rows),
        "window_ensemble_metrics": _write_rows(metrics_dir / "window_ensemble_metrics.csv", ensemble_window_rows),
        "_daily_member_metrics": _write_rows(metrics_dir / "_daily_member_metrics.csv", member_daily_rows),
        "_member_window_metrics": _write_rows(metrics_dir / "_member_window_metrics.csv", member_window_rows),
    }

    _save_metric_boxplots(ensemble_daily_rows, windows=windows_seq, output_dir=plots_dir)

    if make_snapshots:
        for field in fields:
            _save_member_snapshot_figures(
                forecasts_by_experiment["coupled"],
                forecasts_by_experiment["uncoupled"],
                truth,
                field=field,
                snapshot_days=snapshot_days,
                plot_member_indices=plot_members,
                output_dir=plots_dir,
            )
    if make_gifs:
        for field in fields:
            _save_member_gifs(
                forecasts_by_experiment["coupled"],
                forecasts_by_experiment["uncoupled"],
                truth,
                field=field,
                plot_member_indices=plot_members,
                output_dir=plots_dir,
            )

    return outputs


def _longify_case_metric_rows(
    case_dir: Path,
    *,
    init_time: str,
    filename: str,
) -> list[dict[str, Any]]:
    rows = load_long_record_csv(_case_metric_path(case_dir, filename))
    records: list[dict[str, Any]] = []
    for row in rows:
        for metric in ("rmse", "mae", "crps"):
            if metric not in row or row[metric] in ("", None):
                continue
            record = {
                "init_time": init_time,
                "experiment": row["experiment"],
                "field": row["field"],
                "metric": metric,
                "value": float(row[metric]),
            }
            if "lead_day" in row:
                record["lead_day"] = int(row["lead_day"])
            if "window" in row:
                record["window"] = row["window"]
            records.append(record)
    return records


def _longify_member_window_rows(case_dir: Path, *, init_time: str) -> list[dict[str, Any]]:
    rows = load_long_record_csv(_case_metric_path(case_dir, "_member_window_metrics.csv"))
    records: list[dict[str, Any]] = []
    for row in rows:
        for metric in ("rmse", "mae"):
            if metric not in row or row[metric] in ("", None):
                continue
            records.append(
                {
                    "init_time": init_time,
                    "experiment": row["experiment"],
                    "member": int(row["member"]),
                    "field": row["field"],
                    "metric": metric,
                    "window": row["window"],
                    "value": float(row[metric]),
                }
            )
    return records


def postprocess_campaign(
    case_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
    windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]] = DEFAULT_S2S_WINDOWS,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = [Path(case_dir) for case_dir in case_dirs]
    case_ensemble_daily_records: list[dict[str, Any]] = []
    case_ensemble_window_records: list[dict[str, Any]] = []
    case_member_window_records: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        init_time = str(_open_dataset(_prepared_dataset_path(case_dir, "initial.nc"))["time"].values[0])
        case_ensemble_daily_records.extend(
            _longify_case_metric_rows(case_dir, init_time=init_time, filename="daily_ensemble_metrics.csv")
        )
        case_ensemble_window_records.extend(
            _longify_case_metric_rows(case_dir, init_time=init_time, filename="window_ensemble_metrics.csv")
        )
        member_window_path = _case_metric_path(case_dir, "_member_window_metrics.csv")
        if member_window_path.exists():
            case_member_window_records.extend(_longify_member_window_rows(case_dir, init_time=init_time))

    init_mean_ensemble_daily = aggregate_long_records(
        case_ensemble_daily_records,
        group_keys=("experiment", "field", "metric", "lead_day"),
    )
    init_mean_ensemble_window = aggregate_long_records(
        case_ensemble_window_records,
        group_keys=("experiment", "field", "metric", "window"),
    )
    init_mean_member_window = aggregate_long_records(
        case_member_window_records,
        group_keys=("experiment", "member", "field", "metric", "window"),
    ) if case_member_window_records else []

    outputs = {
        "case_ensemble_daily_metrics": _write_rows(metrics_dir / "case_ensemble_daily_metrics.csv", case_ensemble_daily_records),
        "case_ensemble_window_metrics": _write_rows(metrics_dir / "case_ensemble_window_metrics.csv", case_ensemble_window_records),
        "init_mean_ensemble_daily_metrics": _write_rows(metrics_dir / "init_mean_ensemble_daily_metrics.csv", init_mean_ensemble_daily),
        "init_mean_ensemble_window_metrics": _write_rows(metrics_dir / "init_mean_ensemble_window_metrics.csv", init_mean_ensemble_window),
    }
    if case_member_window_records:
        outputs["_case_member_window_metrics"] = _write_rows(
            metrics_dir / "_case_member_window_metrics.csv",
            case_member_window_records,
        )
        outputs["_init_mean_member_window_metrics"] = _write_rows(
            metrics_dir / "_init_mean_member_window_metrics.csv",
            init_mean_member_window,
        )

    _save_campaign_metric_figures(
        case_ensemble_window_records,
        output_dir=plots_dir,
        windows=windows,
    )
    return outputs


def postprocess_case_dirs_center_crps(
    *,
    case_dirs: Sequence[str | Path],
    output_dir: str | Path,
    fields: Sequence[str] = ("t-850", "z-500", "q-700"),
    windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]] = DEFAULT_S2S_WINDOWS,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = [Path(case_dir) for case_dir in case_dirs]
    if not case_dirs:
        raise ValueError("postprocess_case_dirs_center_crps requires at least one case directory")

    case_ensemble_daily_records: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        init_time = str(_open_dataset(_prepared_dataset_path(case_dir, "initial.nc"))["time"].values[0])
        case_ensemble_daily_records.extend(
            _longify_case_metric_rows(case_dir, init_time=init_time, filename="daily_ensemble_metrics.csv")
        )

    init_mean_ensemble_daily = aggregate_long_records(
        case_ensemble_daily_records,
        group_keys=("experiment", "field", "metric", "lead_day"),
    )
    outputs = {
        "case_ensemble_daily_metrics": _write_rows(
            metrics_dir / "case_ensemble_daily_metrics.csv",
            case_ensemble_daily_records,
        ),
        "init_mean_ensemble_daily_metrics": _write_rows(
            metrics_dir / "init_mean_ensemble_daily_metrics.csv",
            init_mean_ensemble_daily,
        ),
    }
    sfno_postprocess_campaign_center_crps(
        campaign_dirs=[output_dir],
        output_dir=output_dir,
        fields=fields,
        windows=_window_map(windows),
    )
    return outputs


def postprocess_campaign_center_crps(
    *,
    campaign_dirs: Sequence[str | Path],
    output_dir: str | Path,
    fields: Sequence[str] = ("t-850", "z-500", "q-700"),
    windows: Sequence[LeadTimeWindow] | Mapping[str, tuple[int, int]] = DEFAULT_S2S_WINDOWS,
) -> None:
    sfno_postprocess_campaign_center_crps(
        campaign_dirs=campaign_dirs,
        output_dir=output_dir,
        fields=fields,
        windows=_window_map(windows),
    )


__all__ = [
    "EXPERIMENTS",
    "FIELD_LABELS",
    "BOX_COLORS",
    "postprocess_campaign",
    "postprocess_campaign_center_crps",
    "postprocess_case",
    "postprocess_case_dirs_center_crps",
]
