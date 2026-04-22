"""Postprocessing for initialized legoESM slab paired forecasts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.regridding import get_cubedsphere_to_latlon_weights
from legoesm.ml.s2s.legoesm_slab.metrics import (
    DEFAULT_S2S_WINDOWS,
    LeadTimeWindow,
    aggregate_long_records,
    compute_forecast_delta_table,
    save_metric_summary_csv,
    save_metric_table_csv,
    score_forecast_metrics,
)
from legoesm.ml.s2s.legoesm_slab.teleconnections import (
    compute_mjo_wind_shear_proxy,
    compute_nino34_series,
)
from legoesm.ml.s2s.legoesm_slab.preparation import (
    PreparationConfig,
    _apply_cubedsphere_to_latlon_masked,
    _blend_cubedsphere_scalar,
    _interp_latlon_to_cubedsphere_masked_2d,
)
from legoesm.ml.s2s.plotting import (
    EXPERIMENT_COLORS,
    normalize_longitude_data,
    plot_latlon_map,
)

EXPERIMENTS = ("coupled", "uncoupled")
FIELD_PLOT_INFO = (
    ("temperature", 850, "T850", 1.0, "viridis"),
    ("geopotential", 500, "Z500", 1.0 / constants.g, "viridis"),
    ("specific_humidity", 700, "Q700", 1.0, "viridis"),
    ("sea_surface_temperature", None, "SST", 1.0, "coolwarm"),
)

PRESSURE_LEVEL_FIELDS = {
    "temperature": 850,
    "geopotential": 500,
    "specific_humidity": 700,
    "u_component_of_wind": 850,
    "v_component_of_wind": 850,
}
FIELD_COLUMN_INFO = {
    "T850": "rmse.temperature_850",
    "Z500": "rmse.geopotential_500",
    "Q700": "rmse.specific_humidity_700",
    "SST": "rmse.sea_surface_temperature",
}


def _open_dataset(path: str | Path) -> xr.Dataset:
    return xr.open_dataset(path)


def _window_sequence(windows: Sequence[LeadTimeWindow]) -> tuple[LeadTimeWindow, ...]:
    return tuple(windows)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _active_windows(
    lead_days: Sequence[int],
    windows: Sequence[LeadTimeWindow],
) -> tuple[LeadTimeWindow, ...]:
    lead = np.asarray(lead_days, dtype=int)
    return tuple(
        window
        for window in windows
        if np.any((lead >= window.start_day) & (lead <= window.end_day))
    )


def _select_plot_field(dataset: xr.Dataset, variable: str, level: int | None, scale: float) -> xr.DataArray:
    field = dataset[variable]
    if level is not None:
        field = field.sel(level=level, method="nearest")
    return field * float(scale)


def _strip_nonspatial_plot_metadata(field: xr.DataArray) -> xr.DataArray:
    """Drop scalar metadata/length-1 dims so map plotting sees a 2-D lat-lon field."""
    extra_coords = [name for name in field.coords if name not in field.dims]
    if extra_coords:
        field = field.reset_coords(extra_coords, drop=True)
    extra_dims = [dim for dim in field.dims if dim not in {"latitude", "longitude"}]
    if extra_dims:
        field = field.squeeze(drop=True)
    return field


def _build_plot_roundtrip_context(
    case_dir: Path,
    sample_field: xr.DataArray,
) -> dict[str, Any] | None:
    """Context for fair truth roundtrip through the model horizontal transform."""
    metadata_path = case_dir / "_prepared" / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resolution = metadata.get("experiment_config", {}).get("grid", {}).get("resolution")
    if resolution is None:
        return None
    resolution = int(resolution)
    grid = create_cubed_sphere(resolution)
    latitude = np.asarray(sample_field["latitude"], dtype=np.float64)
    longitude = np.asarray(sample_field["longitude"], dtype=np.float64)
    return {
        "grid": grid,
        "weights": get_cubedsphere_to_latlon_weights(
            grid.n,
            n_lon=longitude.size,
            n_lat=latitude.size,
        ),
        "latitude": latitude,
        "longitude": longitude,
        "prep_config": PreparationConfig(),
    }


def _roundtrip_plot_field(
    field: xr.DataArray,
    *,
    context: Mapping[str, Any] | None,
) -> xr.DataArray:
    """Roundtrip a 2-D lat-lon field through cubed-sphere and back for plotting."""
    plot_field = _strip_nonspatial_plot_metadata(normalize_longitude_data(field))
    if context is None:
        return plot_field
    source_lat = np.asarray(plot_field["latitude"], dtype=np.float64)
    source_lon = np.asarray(plot_field["longitude"], dtype=np.float64)
    field_cs = _interp_latlon_to_cubedsphere_masked_2d(
        plot_field,
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat_rad=context["grid"].grid_lat,
        target_lon_rad=context["grid"].grid_lon,
        config=context["prep_config"],
    )
    field_cs = _blend_cubedsphere_scalar(field_cs, config=context["prep_config"])
    roundtripped = _apply_cubedsphere_to_latlon_masked(field_cs, context["weights"]).astype(np.float32)
    return xr.DataArray(
        roundtripped,
        coords={"latitude": context["latitude"], "longitude": context["longitude"]},
        dims=("latitude", "longitude"),
    )


def _roundtrip_plot_mask(
    field: xr.DataArray,
    *,
    context: Mapping[str, Any] | None,
    threshold: float = 0.5,
) -> xr.DataArray:
    """Roundtrip a finite-value mask through the same horizontal transform."""
    plot_field = _strip_nonspatial_plot_metadata(normalize_longitude_data(field))
    valid = xr.DataArray(
        np.isfinite(np.asarray(plot_field, dtype=float)).astype(np.float32),
        coords=plot_field.coords,
        dims=plot_field.dims,
    )
    if context is None:
        return valid > float(threshold)
    source_lat = np.asarray(valid["latitude"], dtype=np.float64)
    source_lon = np.asarray(valid["longitude"], dtype=np.float64)
    valid_cs = _interp_latlon_to_cubedsphere_masked_2d(
        valid,
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat_rad=context["grid"].grid_lat,
        target_lon_rad=context["grid"].grid_lon,
        config=context["prep_config"],
    )
    valid_cs = _blend_cubedsphere_scalar(valid_cs, config=context["prep_config"])
    valid_rt = _apply_cubedsphere_to_latlon_masked(valid_cs, context["weights"]).astype(np.float32)
    return xr.DataArray(
        valid_rt > float(threshold),
        coords={"latitude": context["latitude"], "longitude": context["longitude"]},
        dims=("latitude", "longitude"),
    )


def _apply_pressure_level_masks(dataset: xr.Dataset) -> xr.Dataset:
    """Mask pressure-level fields where the target pressure lies below surface."""
    if "surface_pressure" not in dataset:
        return dataset
    masked = dataset.copy()
    surface_pressure = masked["surface_pressure"]
    for variable, level_hpa in PRESSURE_LEVEL_FIELDS.items():
        if variable not in masked:
            continue
        field = masked[variable]
        level_mask = surface_pressure
        if "lead_day" in field.dims and "lead_day" not in level_mask.dims and "lead_day" in level_mask.coords:
            level_mask = level_mask.swap_dims({"time": "lead_day"})
            if "time" in level_mask.coords:
                level_mask = level_mask.reset_coords("time", drop=True)
        elif "time" in field.dims and "time" not in level_mask.dims and "time" in level_mask.coords:
            level_mask = level_mask.swap_dims({"lead_day": "time"})
            if "lead_day" in level_mask.coords:
                level_mask = level_mask.reset_coords("lead_day", drop=True)
        level_mask = level_mask >= float(level_hpa) * 100.0
        if "level" in field.dims:
            field = field.where(field["level"] != level_hpa, field.where(level_mask))
        else:
            field = field.where(level_mask)
        masked[variable] = field
    return masked


def _fill_plot_gaps(field: xr.DataArray) -> xr.DataArray:
    """Fill NaN holes for plotting only, leaving raw data/metrics unchanged."""
    plot_field = _strip_nonspatial_plot_metadata(normalize_longitude_data(field))
    values = np.asarray(plot_field, dtype=np.float32)
    if values.ndim != 2:
        return plot_field
    valid = np.isfinite(values)
    if valid.all() or not valid.any():
        return plot_field
    yy, xx = np.indices(values.shape)
    valid_yx = np.column_stack(np.where(valid))
    missing_yx = np.column_stack(np.where(~valid))
    filled = values.copy()
    for y, x in missing_yx:
        dist2 = (valid_yx[:, 0] - y) ** 2 + (valid_yx[:, 1] - x) ** 2
        nearest = valid_yx[int(np.argmin(dist2))]
        filled[y, x] = values[nearest[0], nearest[1]]
    return xr.DataArray(filled, coords=plot_field.coords, dims=plot_field.dims)


def _save_timeseries_plot(
    metric_tables: Mapping[str, Mapping[str, np.ndarray]],
    *,
    lead_days: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, column_name) in zip(axes.ravel(), FIELD_COLUMN_INFO.items(), strict=True):
        for experiment, table in metric_tables.items():
            if column_name not in table:
                continue
            ax.plot(
                lead_days,
                table[column_name],
                label=experiment,
                color=EXPERIMENT_COLORS[experiment],
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Lead day")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_snapshot_maps(
    initial: xr.Dataset,
    forcing: xr.Dataset | None,
    truth: xr.Dataset,
    forecasts: Mapping[str, xr.Dataset],
    *,
    case_dir: Path,
    snapshot_days: Sequence[int],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    available_experiments = tuple(forecasts.keys())
    if not available_experiments:
        return written
    reference = forecasts[available_experiments[0]]
    available_days = set(int(day) for day in reference["lead_day"].values.tolist())
    roundtrip_context = _build_plot_roundtrip_context(
        case_dir,
        _select_plot_field(truth.rename({"time": "lead_day"}), "temperature", 850, 1.0).isel(lead_day=0),
    )
    for day in snapshot_days:
        if int(day) not in available_days:
            continue
        lead_index = int(np.where(reference["lead_day"].values == int(day))[0][0])
        ncols = 2 + len(available_experiments)
        fig, axes = plt.subplots(
            len(FIELD_PLOT_INFO),
            ncols,
            figsize=(5 * ncols, 4 * len(FIELD_PLOT_INFO)),
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        for row_index, (variable, level, title, scale, cmap) in enumerate(FIELD_PLOT_INFO):
            initial_field = _select_plot_field(initial, variable, level, scale)
            if "time" in initial_field.dims:
                initial_field = initial_field.isel(time=0, drop=True)
            truth_field_raw = _select_plot_field(
                truth.rename({"time": "lead_day"}), variable, level, scale
            ).isel(lead_day=lead_index)
            initial_field = _roundtrip_plot_field(initial_field, context=None)
            truth_field = _roundtrip_plot_field(truth_field_raw, context=roundtrip_context)
            if variable == "sea_surface_temperature":
                ocean_mask = _roundtrip_plot_mask(
                    truth_field_raw,
                    context=roundtrip_context,
                )
                initial_field = initial_field.where(ocean_mask)
                truth_field = truth_field.where(ocean_mask)
            else:
                initial_field = _fill_plot_gaps(initial_field)
                truth_field = _fill_plot_gaps(truth_field)
            # Anchor the absolute color scale on the truth field so forecast
            # outliers do not wash out the structure in the reference panel.
            vmin = float(np.nanmin(np.asarray(truth_field, dtype=float)))
            vmax = float(np.nanmax(np.asarray(truth_field, dtype=float)))
            plot_latlon_map(axes[row_index, 0], initial_field, f"Initial State {title}", vmin=vmin, vmax=vmax, cmap=cmap)
            plot_latlon_map(axes[row_index, 1], truth_field, f"Truth {title}", vmin=vmin, vmax=vmax, cmap=cmap)
            for col_offset, experiment in enumerate(available_experiments, start=2):
                forecast_field = _select_plot_field(forecasts[experiment], variable, level, scale).isel(lead_day=lead_index)
                if variable == "sea_surface_temperature":
                    forecast_field = normalize_longitude_data(forecast_field)
                    forecast_field = forecast_field.where(ocean_mask)
                forecast_field = _strip_nonspatial_plot_metadata(forecast_field)
                if variable != "sea_surface_temperature":
                    forecast_field = _fill_plot_gaps(forecast_field)
                plot_latlon_map(
                    axes[row_index, col_offset],
                    forecast_field,
                    f"{experiment.replace('_', ' ').title()} {title}",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
        fig.suptitle(f"Lead day {int(day):03d}", fontsize=14)
        path = output_dir / f"maps_day{int(day):03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def _teleconnection_rows(
    truth: xr.Dataset,
    forecasts: Mapping[str, xr.Dataset],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enso_rows: list[dict[str, Any]] = []
    shear_rows: list[dict[str, Any]] = []
    truth_time = truth["time"].values

    truth_nino = compute_nino34_series(truth)
    for time_index, time_value in enumerate(truth_time):
        enso_rows.append(
            {
                "experiment": "truth",
                "lead_day": int(time_index + 1),
                "time": str(np.datetime64(time_value, "s")),
                "nino34_sst": float(truth_nino.isel(time=time_index)),
            }
        )

    for experiment, forecast in forecasts.items():
        nino = compute_nino34_series(forecast)
        shear = compute_mjo_wind_shear_proxy(forecast)
        for lead_index, lead_day in enumerate(forecast["lead_day"].values.tolist()):
            timestamp = forecast["time"].isel(lead_day=lead_index).item()
            enso_rows.append(
                {
                    "experiment": experiment,
                    "lead_day": int(lead_day),
                    "time": str(np.datetime64(timestamp, "s")),
                    "nino34_sst": float(nino.isel(lead_day=lead_index)),
                }
            )
            shear_rows.append(
                {
                    "experiment": experiment,
                    "lead_day": int(lead_day),
                    "time": str(np.datetime64(timestamp, "s")),
                    "mjo_wind_shear_proxy": float(shear.isel(lead_day=lead_index)),
                }
            )
    return enso_rows, shear_rows


def postprocess_case(
    case_dir: str | Path,
    *,
    windows: Sequence[LeadTimeWindow] = DEFAULT_S2S_WINDOWS,
    snapshot_days: Sequence[int] = (1, 15, 29, 42),
) -> dict[str, Path]:
    """Score and plot one paired legoESM slab case."""
    case_dir = Path(case_dir)
    metrics_dir = case_dir / "metrics"
    plots_dir = case_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    truth = normalize_longitude_data(_apply_pressure_level_masks(_open_dataset(case_dir / "_prepared" / "truth.nc")))
    if "lead_day" in truth.coords and "time" in truth.coords:
        truth = truth.drop_vars("lead_day")
    initial_path = case_dir / "_prepared" / "initial.nc"
    if initial_path.exists():
        initial = _open_dataset(initial_path)
    else:
        initial = truth.isel(time=0, drop=False)
    initial = normalize_longitude_data(_apply_pressure_level_masks(initial))
    forcing_candidates = sorted((case_dir / "_prepared").glob("surface_forcing*.nc"))
    forcing = (
        normalize_longitude_data(_open_dataset(forcing_candidates[0]))
        if forcing_candidates
        else None
    )
    forecasts = {
        experiment: normalize_longitude_data(_apply_pressure_level_masks(_open_dataset(case_dir / experiment / "forecast.nc")))
        for experiment in EXPERIMENTS
        if (case_dir / experiment / "forecast.nc").exists()
    }
    if not forecasts:
        raise FileNotFoundError(
            f"No forecast.nc files found under {case_dir} for experiments {EXPERIMENTS!r}"
        )
    reference_experiment = next(iter(forecasts))
    lead_days = np.asarray(forecasts[reference_experiment]["lead_day"], dtype=int)
    n_forecast = int(lead_days.size)
    if "time" in truth.dims:
        n_truth = int(truth.sizes["time"])
        if n_truth == n_forecast + 1:
            truth_forecast = truth.isel(time=slice(1, None))
        else:
            truth_forecast = truth.isel(time=slice(0, n_forecast))
    else:
        truth_forecast = truth
    if "sea_surface_temperature" in truth_forecast:
        truth_sst = normalize_longitude_data(truth_forecast["sea_surface_temperature"])
        sst_mask = truth_sst
        if "time" in sst_mask.dims:
            sst_mask = sst_mask.rename({"time": "lead_day"}).assign_coords(
                lead_day=forecasts[reference_experiment]["lead_day"].values
            )
        sst_mask = sst_mask.notnull()
        for experiment, forecast in tuple(forecasts.items()):
            if "sea_surface_temperature" not in forecast:
                continue
            forecast_sst = normalize_longitude_data(forecast["sea_surface_temperature"])
            forecasts[experiment]["sea_surface_temperature"] = forecast_sst.where(sst_mask)
    windows_seq = _active_windows(lead_days, _window_sequence(windows))

    outputs: dict[str, Path] = {}
    daily_tables: dict[str, Mapping[str, np.ndarray]] = {}
    experiment_summaries: dict[str, dict[str, float]] = {}
    daily_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for experiment in forecasts:
        daily_table, summary = score_forecast_metrics(
            forecasts[experiment],
            truth_forecast,
            windows=windows_seq,
        )
        daily_tables[experiment] = daily_table
        lead_days = np.asarray(forecasts[experiment]["lead_day"], dtype=int)
        outputs[f"daily_{experiment}_metrics"] = save_metric_table_csv(
            daily_table,
            metrics_dir / f"daily_{experiment}_metrics.csv",
            lead_days=lead_days,
        )
        outputs[f"window_{experiment}_metrics"] = save_metric_summary_csv(
            summary,
            metrics_dir / f"window_{experiment}_metrics.csv",
        )
        experiment_summaries[experiment] = {
            f"{window_name}.{metric_name}": float(value)
            for window_name, metric_values in summary.items()
            for metric_name, value in metric_values.items()
        }
        for row_index, lead_day in enumerate(lead_days.tolist()):
            for name, values in daily_table.items():
                daily_rows.append(
                    {
                        "experiment": experiment,
                        "lead_day": int(lead_day),
                        "metric": name,
                        "value": float(values[row_index]),
                    }
                )
        for window_name, metric_values in summary.items():
            for name, value in metric_values.items():
                window_rows.append(
                    {
                        "experiment": experiment,
                        "window": window_name,
                        "metric": name,
                        "value": float(value),
                    }
                )

    delta_summary: dict[str, dict[str, float]] = {}
    if "coupled" in forecasts and "uncoupled" in forecasts:
        delta_daily = compute_forecast_delta_table(forecasts["coupled"], forecasts["uncoupled"])
        delta_summary = {
            window.name: {
                name: float(np.nanmean(np.asarray(values, dtype=float)[(lead_days >= window.start_day) & (lead_days <= window.end_day)]))
                for name, values in delta_daily.items()
            }
            for window in windows_seq
        }
        outputs["daily_coupled_uncoupled_delta"] = save_metric_table_csv(
            delta_daily,
            metrics_dir / "daily_coupled_uncoupled_delta.csv",
            lead_days=lead_days,
        )
        outputs["window_coupled_uncoupled_delta"] = save_metric_summary_csv(
            delta_summary,
            metrics_dir / "window_coupled_uncoupled_delta.csv",
        )
    outputs["case_metric_rows"] = _write_rows(metrics_dir / "case_metric_rows.csv", daily_rows)
    outputs["case_window_rows"] = _write_rows(metrics_dir / "case_window_rows.csv", window_rows)
    outputs["headline_rmse_timeseries"] = _save_timeseries_plot(
        daily_tables,
        lead_days=lead_days,
        output_path=plots_dir / "headline_rmse_timeseries.png",
    )
    written_maps = _save_snapshot_maps(
        initial,
        forcing,
        truth_forecast,
        forecasts,
        case_dir=case_dir,
        snapshot_days=snapshot_days,
        output_dir=plots_dir,
    )
    if written_maps:
        outputs["snapshot_maps"] = written_maps[0]

    enso_rows, shear_rows = _teleconnection_rows(truth_forecast, forecasts)
    outputs["enso_series"] = _write_rows(metrics_dir / "enso_nino34_series.csv", enso_rows)
    outputs["mjo_proxy_series"] = _write_rows(metrics_dir / "mjo_wind_shear_proxy.csv", shear_rows)

    summary_payload = {
        "experiments": experiment_summaries,
        "coupled_uncoupled_delta": {window: values for window, values in delta_summary.items()},
        "run_summary": json.loads((case_dir / "run_summary.json").read_text(encoding="utf-8"))
        if (case_dir / "run_summary.json").exists() else {},
    }
    outputs["summary_json"] = metrics_dir / "summary.json"
    outputs["summary_json"].write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    initial.close()
    if forcing is not None:
        forcing.close()
    truth_forecast.close()
    truth.close()
    for dataset in forecasts.values():
        dataset.close()
    return outputs


def _load_metric_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def postprocess_campaign(
    case_dirs: Sequence[str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Aggregate windowed deterministic scores across prepared case directories."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    case_window_rows: list[dict[str, Any]] = []
    for case_dir in (Path(case_dir) for case_dir in case_dirs):
        for experiment in EXPERIMENTS:
            for row in _load_metric_summary_rows(case_dir / "metrics" / f"window_{experiment}_metrics.csv"):
                window_name = row["window"]
                for metric_name, value in row.items():
                    if metric_name == "window":
                        continue
                    case_window_rows.append(
                        {
                            "case": case_dir.name,
                            "experiment": experiment,
                            "window": window_name,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )

    init_mean_rows = aggregate_long_records(
        case_window_rows,
        group_keys=("experiment", "window", "metric"),
    )
    outputs = {
        "case_window_metrics": _write_rows(metrics_dir / "case_window_metrics.csv", case_window_rows),
        "init_mean_window_metrics": _write_rows(metrics_dir / "init_mean_window_metrics.csv", init_mean_rows),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, metric_name) in zip(axes.ravel(), FIELD_COLUMN_INFO.items(), strict=True):
        for experiment in EXPERIMENTS:
            rows = [
                row for row in init_mean_rows
                if row["experiment"] == experiment and row["metric"] == metric_name
            ]
            if not rows:
                continue
            ax.plot(
                [row["window"] for row in rows],
                [row["value"] for row in rows],
                label=experiment,
                color=EXPERIMENT_COLORS[experiment],
                marker="o",
            )
        ax.set_title(title)
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
    campaign_plot = plots_dir / "campaign_window_rmse.png"
    fig.savefig(campaign_plot, dpi=150)
    plt.close(fig)
    outputs["campaign_window_rmse_plot"] = campaign_plot
    return outputs


__all__ = ["EXPERIMENTS", "FIELD_PLOT_INFO", "postprocess_campaign", "postprocess_case"]
