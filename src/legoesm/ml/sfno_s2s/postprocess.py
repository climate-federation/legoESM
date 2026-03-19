"""Ensemble postprocessing and plotting helpers for SFNO S2S rollouts."""

from __future__ import annotations

import csv
from io import BytesIO
from io import StringIO
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from legoesm.ml.sfno_s2s.evaluation import (
    aggregate_long_records,
    compute_ensemble_daily_metrics,
    summarize_window_metrics,
    write_metric_rows,
)


ATMOSPHERE_FIELDS = {
    "t850": ("t-850", "T850", "K"),
    "z500": ("z-500", "Z500", "m"),
    "q700": ("q-700", "Q700", "kg kg-1"),
}

CAMPAIGN_FIELD_SPECS = {
    "t-850": ("T850", "t850_center_crps_comparison.png"),
    "z-500": ("Z500", "z500_center_crps_comparison.png"),
    "q-700": ("Q700", "q700_center_crps_comparison.png"),
}

CENTER_URLS = {
    "ECMWF": "https://huggingface.co/datasets/LEAP/ChaosBench/raw/main/logs/ecmwf_ensemble/eval/crps_ecmwf.csv",
    "UKMO": "https://huggingface.co/datasets/LEAP/ChaosBench/raw/main/logs/ukmo_ensemble/eval/crps_ukmo.csv",
    "NCEP": "https://huggingface.co/datasets/LEAP/ChaosBench/raw/main/logs/ncep_ensemble/eval/crps_ncep.csv",
    "CMA": "https://huggingface.co/datasets/LEAP/ChaosBench/raw/main/logs/cma_ensemble/eval/crps_cma.csv",
}

CENTER_COLORS = {
    "ECMWF": "#6c757d",
    "UKMO": "#868e96",
    "NCEP": "#adb5bd",
    "CMA": "#ced4da",
    "SFNO (coupled)": "#1f77b4",
    "SFNO (uncoupled)": "#ff7f0e",
}


def open_rollout_members(paths: Sequence[str | Path]) -> list[xr.Dataset]:
    """Open a collection of rollout NetCDF files."""
    return [xr.open_dataset(Path(path)) for path in paths]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _channel_index(ds: xr.Dataset, label: str) -> int:
    labels = [str(value) for value in ds["channel"].values.tolist()]
    return labels.index(label)


def _plot_map(ax: plt.Axes, field: xr.DataArray, title: str, *, vmin: float, vmax: float) -> None:
    plot_field = field.transpose("latitude", "longitude")
    plot_field.plot(
        ax=ax,
        x="longitude",
        y="latitude",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _frame_to_image(fig: plt.Figure) -> np.ndarray:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def _sst_truth(ds: xr.Dataset) -> xr.DataArray:
    if "target_sea_surface_temperature" in ds:
        return ds["target_sea_surface_temperature"]
    if "target_forcing" in ds and "forcing_channel" in ds.coords:
        labels = [str(value) for value in ds["forcing_channel"].values.tolist()]
        if "sosstsst" in labels:
            return ds["target_forcing"].isel(forcing_channel=labels.index("sosstsst"))
    raise KeyError("No SST truth field available in rollout dataset.")


def _draw_window_boxplots(
    ax: plt.Axes,
    *,
    rows: list[dict[str, object]],
    field: str,
    metric: str,
    windows: dict[str, tuple[int, int]],
) -> None:
    window_names = list(windows.keys())
    offsets = {"coupled": -0.18, "uncoupled": 0.18}
    colors = {"coupled": "#1f77b4", "uncoupled": "#ff7f0e"}
    positions: list[float] = []
    data: list[np.ndarray] = []
    facecolors: list[str] = []
    labels: list[str] = []

    for window_index, window_name in enumerate(window_names):
        start_day, end_day = windows[window_name]
        center = float(window_index)
        for experiment in ("coupled", "uncoupled"):
            selected = [
                float(record[metric])
                for record in rows
                if record["field"] == field
                and record["experiment"] == experiment
                and start_day <= int(record["lead_day"]) <= end_day
            ]
            if not selected:
                continue
            positions.append(center + offsets[experiment])
            data.append(np.asarray(selected, dtype=float))
            facecolors.append(colors[experiment])
            labels.append(experiment)

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
    ax.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color=colors["coupled"], lw=8, label="coupled"),
        plt.Line2D([0], [0], color=colors["uncoupled"], lw=8, label="uncoupled"),
    ]
    ax.legend(handles=handles)


def _save_metric_boxplots(
    daily_rows: list[dict[str, object]],
    *,
    windows: dict[str, tuple[int, int]],
    output_dir: Path,
) -> None:
    metric_fields = [("t-850", "t850"), ("z-500", "z500"), ("q-700", "q700"), ("sosstsst", "sst")]
    title_map = {"t850": "T850", "z500": "Z500", "q700": "Q700", "sst": "SST"}
    for field, stem in metric_fields:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        _draw_window_boxplots(axes[0], rows=daily_rows, field=field, metric="rmse", windows=windows)
        axes[0].set_title(f"{title_map[stem]} RMSE")
        _draw_window_boxplots(axes[1], rows=daily_rows, field=field, metric="crps", windows=windows)
        axes[1].set_title(f"{title_map[stem]} CRPS")
        fig.savefig(output_dir / f"{stem}_metric_summary.png", dpi=150)
        plt.close(fig)


def _fetch_center_series(url: str, field: str) -> np.ndarray:
    with urlopen(url) as response:
        text = response.read().decode("utf-8")
    rows = list(csv.DictReader(StringIO(text)))
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def _load_campaign_daily_crps(campaign_dirs: Sequence[str | Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for campaign_dir in campaign_dirs:
        path = Path(campaign_dir) / "metrics" / "case_ensemble_daily_metrics.csv"
        for row in _read_csv(path):
            if row["metric"] != "crps":
                continue
            records.append(
                {
                    "init_time": row["init_time"],
                    "experiment": row["experiment"],
                    "field": row["field"],
                    "metric": row["metric"],
                    "lead_day": int(row["lead_day"]),
                    "value": float(row["value"]),
                }
            )
    return aggregate_long_records(
        records,
        group_keys=("experiment", "field", "metric", "lead_day"),
    )


def _window_values(series: np.ndarray, start_day: int, end_day: int) -> np.ndarray:
    return series[start_day - 1 : end_day]


def _write_center_summary_rows(
    *,
    output_path: Path,
    field: str,
    center_series: dict[str, np.ndarray],
    sfno_rows: list[dict[str, object]],
    windows: dict[str, tuple[int, int]],
) -> None:
    rows: list[dict[str, object]] = []
    for center, series in center_series.items():
        for window_name, (start_day, end_day) in windows.items():
            values = _window_values(series, start_day, end_day)
            rows.append(
                {
                    "field": field,
                    "source": center,
                    "window": window_name,
                    "mean_crps": float(np.mean(values)),
                }
            )

    for experiment in ("coupled", "uncoupled"):
        experiment_rows = [
            row for row in sfno_rows
            if row["field"] == field and row["experiment"] == experiment
        ]
        series = np.asarray([float(row["value"]) for row in experiment_rows], dtype=float)
        for window_name, (start_day, end_day) in windows.items():
            values = _window_values(series, start_day, end_day)
            rows.append(
                {
                    "field": field,
                    "source": f"SFNO ({experiment})",
                    "window": window_name,
                    "mean_crps": float(np.mean(values)),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("field", "source", "window", "mean_crps"))
        writer.writeheader()
        writer.writerows(rows)


def _plot_campaign_center_crps_field(
    *,
    field: str,
    title: str,
    center_series: dict[str, np.ndarray],
    sfno_rows: list[dict[str, object]],
    windows: dict[str, tuple[int, int]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(windows), figsize=(6.5 * len(windows), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)

    sfno_by_experiment: dict[str, np.ndarray] = {}
    for experiment in ("coupled", "uncoupled"):
        experiment_rows = [
            row for row in sfno_rows
            if row["field"] == field and row["experiment"] == experiment
        ]
        sfno_by_experiment[experiment] = np.asarray(
            [float(row["value"]) for row in experiment_rows],
            dtype=float,
        )

    for ax, (window_name, (start_day, end_day)) in zip(axes, windows.items(), strict=True):
        entries: list[tuple[str, np.ndarray, float]] = []
        for center, series in center_series.items():
            values = _window_values(series, start_day, end_day)
            entries.append((center, values, float(np.mean(values))))
        for experiment, series in sfno_by_experiment.items():
            values = _window_values(series, start_day, end_day)
            label = f"SFNO ({experiment})"
            entries.append((label, values, float(np.mean(values))))

        entries.sort(key=lambda item: item[2], reverse=True)
        labels = [item[0] for item in entries]
        values = [item[1] for item in entries]
        means = [item[2] for item in entries]

        artists = ax.boxplot(values, patch_artist=True, showfliers=False)
        for patch, label in zip(artists["boxes"], labels, strict=True):
            patch.set_facecolor(CENTER_COLORS[label])
            patch.set_alpha(0.8)
        for median in artists["medians"]:
            median.set_color("black")
        ax.set_xticks(range(1, len(labels) + 1), labels=labels, rotation=25, ha="right")
        ax.set_ylabel("CRPS")
        ax.set_title(f"{title} {window_name}")
        ax.grid(True, axis="y", alpha=0.3)
        for idx, mean in enumerate(means, start=1):
            ax.text(idx, mean, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_member_snapshot_figures(
    coupled_members: Sequence[xr.Dataset],
    uncoupled_members: Sequence[xr.Dataset],
    *,
    field_key: str,
    snapshot_days: Sequence[int],
    plot_member_indices: Sequence[int],
    output_dir: Path,
) -> None:
    if field_key == "sst":
        field_getter = lambda ds, lead_idx: ds["sea_surface_temperature"].isel(lead_day=lead_idx)
        truth_getter = lambda ds, lead_idx: _sst_truth(ds).isel(lead_day=lead_idx)
        title_prefix = "SST"
        colorbar_label = "degC"
    else:
        channel_label, title_prefix, colorbar_label = ATMOSPHERE_FIELDS[field_key]
        field_getter = lambda ds, lead_idx: ds["prediction"].isel(lead_day=lead_idx, channel=_channel_index(ds, channel_label))
        truth_getter = lambda ds, lead_idx: ds["target"].isel(lead_day=lead_idx, channel=_channel_index(ds, channel_label))

    reference = coupled_members[0]
    lead_to_index = {int(day): i for i, day in enumerate(reference["lead_day"].values.tolist())}

    truth_fields: list[np.ndarray] = []
    for member_index in plot_member_indices:
        coupled = coupled_members[member_index]
        for day in snapshot_days:
            lead_idx = lead_to_index[day]
            truth_fields.append(np.asarray(truth_getter(coupled, lead_idx).values, dtype=float))
    vmin = float(np.nanmin(truth_fields))
    vmax = float(np.nanmax(truth_fields))

    legacy_path = output_dir / f"{field_key}_member_snapshots.png"
    if legacy_path.exists():
        legacy_path.unlink()

    for member_index in plot_member_indices:
        coupled = coupled_members[member_index]
        uncoupled = uncoupled_members[member_index]
        fig, axes = plt.subplots(len(snapshot_days), 3, figsize=(18, 4 * len(snapshot_days)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for offset, day in enumerate(snapshot_days):
            lead_idx = lead_to_index[day]
            truth_field = truth_getter(coupled, lead_idx)
            coupled_field = field_getter(coupled, lead_idx)
            uncoupled_field = field_getter(uncoupled, lead_idx)
            _plot_map(axes[offset, 0], truth_field, f"Truth {title_prefix}, day {day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[offset, 1], coupled_field, f"Coupled {title_prefix}, day {day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[offset, 2], uncoupled_field, f"Uncoupled {title_prefix}, day {day}", vmin=vmin, vmax=vmax)
        fig.suptitle(f"Member {member_index:02d}", fontsize=14)
        fig.colorbar(axes[-1, 2].collections[0], ax=axes, shrink=0.96, label=colorbar_label)
        fig.savefig(output_dir / f"{field_key}_member{member_index:02d}_snapshots.png", dpi=150)
        plt.close(fig)


def _save_member_gifs(
    coupled_members: Sequence[xr.Dataset],
    uncoupled_members: Sequence[xr.Dataset],
    *,
    field_key: str,
    plot_member_indices: Sequence[int],
    output_dir: Path,
    duration: float = 0.45,
) -> None:
    if field_key == "sst":
        field_getter = lambda ds, lead_idx: ds["sea_surface_temperature"].isel(lead_day=lead_idx)
        truth_getter = lambda ds, lead_idx: _sst_truth(ds).isel(lead_day=lead_idx)
        title_prefix = "SST"
        colorbar_label = "degC"
    else:
        channel_label, title_prefix, colorbar_label = ATMOSPHERE_FIELDS[field_key]
        field_getter = lambda ds, lead_idx: ds["prediction"].isel(lead_day=lead_idx, channel=_channel_index(ds, channel_label))
        truth_getter = lambda ds, lead_idx: ds["target"].isel(lead_day=lead_idx, channel=_channel_index(ds, channel_label))

    reference = coupled_members[0]
    lead_days = [int(day) for day in reference["lead_day"].values.tolist()]
    truth_all = []
    for member_index in plot_member_indices:
        coupled = coupled_members[member_index]
        for lead_idx in range(len(lead_days)):
            truth_all.append(np.asarray(truth_getter(coupled, lead_idx).values, dtype=float))
    vmin = float(np.nanmin(truth_all))
    vmax = float(np.nanmax(truth_all))

    for member_index in plot_member_indices:
        coupled = coupled_members[member_index]
        uncoupled = uncoupled_members[member_index]
        frames: list[np.ndarray] = []
        for lead_idx, lead_day in enumerate(lead_days):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            truth_field = truth_getter(coupled, lead_idx)
            coupled_field = field_getter(coupled, lead_idx)
            uncoupled_field = field_getter(uncoupled, lead_idx)
            _plot_map(axes[0], truth_field, f"Truth {title_prefix}, day {lead_day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[1], coupled_field, f"Coupled {title_prefix}, day {lead_day}", vmin=vmin, vmax=vmax)
            _plot_map(axes[2], uncoupled_field, f"Uncoupled {title_prefix}, day {lead_day}", vmin=vmin, vmax=vmax)
            fig.colorbar(axes[2].collections[0], ax=axes, shrink=0.9, label=colorbar_label)
            fig.suptitle(f"Member {member_index:02d}", fontsize=12)
            frames.append(_frame_to_image(fig))
        imageio.mimsave(
            output_dir / f"{field_key}_member{member_index:02d}_daily.gif",
            frames,
            duration=duration,
            loop=0,
        )


def postprocess_ensemble_rollouts(
    *,
    coupled_paths: Sequence[str | Path],
    uncoupled_paths: Sequence[str | Path],
    output_dir: str | Path,
    fields: Sequence[str],
    windows: dict[str, tuple[int, int]],
    snapshot_days: Sequence[int],
    plot_member_indices: Sequence[int],
) -> None:
    """Compute ensemble metrics and representative-member plots from rollout files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    coupled_members = open_rollout_members(coupled_paths)
    uncoupled_members = open_rollout_members(uncoupled_paths)
    if len(coupled_members) != len(uncoupled_members):
        raise ValueError("Coupled and uncoupled member counts must match.")
    if not coupled_members:
        raise ValueError("At least one member is required.")
    for member_index in plot_member_indices:
        if member_index < 0 or member_index >= len(coupled_members):
            raise IndexError(f"Plot member index {member_index} is out of range for {len(coupled_members)} members.")

    coupled_daily = compute_ensemble_daily_metrics(coupled_members, fields=list(fields))
    uncoupled_daily = compute_ensemble_daily_metrics(uncoupled_members, fields=list(fields))
    coupled_windows = summarize_window_metrics(coupled_daily, windows=windows)
    uncoupled_windows = summarize_window_metrics(uncoupled_daily, windows=windows)

    for record in coupled_daily:
        record["experiment"] = "coupled"
    for record in uncoupled_daily:
        record["experiment"] = "uncoupled"
    daily_rows = coupled_daily + uncoupled_daily

    for record in coupled_windows:
        record["experiment"] = "coupled"
    for record in uncoupled_windows:
        record["experiment"] = "uncoupled"
    window_rows = coupled_windows + uncoupled_windows

    write_metric_rows(daily_rows, metrics_dir / "daily_ensemble_metrics.csv")
    write_metric_rows(window_rows, metrics_dir / "window_ensemble_metrics.csv")

    _save_metric_boxplots(daily_rows, windows=windows, output_dir=plots_dir)
    _save_member_snapshot_figures(
        coupled_members,
        uncoupled_members,
        field_key="sst",
        snapshot_days=snapshot_days,
        plot_member_indices=plot_member_indices,
        output_dir=plots_dir,
    )
    _save_member_gifs(
        coupled_members,
        uncoupled_members,
        field_key="sst",
        plot_member_indices=plot_member_indices,
        output_dir=plots_dir,
    )
    for field_key in ATMOSPHERE_FIELDS:
        _save_member_snapshot_figures(
            coupled_members,
            uncoupled_members,
            field_key=field_key,
            snapshot_days=snapshot_days,
            plot_member_indices=plot_member_indices,
            output_dir=plots_dir,
        )
        _save_member_gifs(
            coupled_members,
            uncoupled_members,
            field_key=field_key,
            plot_member_indices=plot_member_indices,
            output_dir=plots_dir,
        )


def postprocess_campaign_center_crps(
    *,
    campaign_dirs: Sequence[str | Path],
    output_dir: str | Path,
    fields: Sequence[str],
    windows: dict[str, tuple[int, int]],
) -> None:
    """Compare aggregated SFNO campaign CRPS against ChaosBench center baselines."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sfno_rows = _load_campaign_daily_crps(campaign_dirs)
    for field in fields:
        if field not in CAMPAIGN_FIELD_SPECS:
            raise ValueError(f"Unsupported field for center comparison: {field}")
        title, filename = CAMPAIGN_FIELD_SPECS[field]
        center_series = {
            center: _fetch_center_series(url, field)
            for center, url in CENTER_URLS.items()
        }
        _write_center_summary_rows(
            output_path=metrics_dir / f"{field.replace('-', '')}_center_window_crps.csv",
            field=field,
            center_series=center_series,
            sfno_rows=sfno_rows,
            windows=windows,
        )
        _plot_campaign_center_crps_field(
            field=field,
            title=title,
            center_series=center_series,
            sfno_rows=sfno_rows,
            windows=windows,
            output_path=plots_dir / filename,
        )


__all__ = [
    "open_rollout_members",
    "postprocess_campaign_center_crps",
    "postprocess_ensemble_rollouts",
]
