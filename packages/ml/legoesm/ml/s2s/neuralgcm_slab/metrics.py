"""Forecast-metric helpers for NeuralGCM slab-ocean experiments."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

from legoesm.ml.s2s.neuralgcm_slab.evaluation import (
    DEFAULT_S2S_WINDOWS,
    LeadTimeWindow,
    aggregate_metric_table,
)


@dataclass(frozen=True)
class FieldSpec:
    """Field specification for forecast verification."""

    name: str
    variable: str
    level: int | None = None
    level_name: str = "level"


MetricFunction = Callable[[xr.DataArray, xr.DataArray], float]


@dataclass(frozen=True)
class MetricSpec:
    """Metric specification for forecast verification."""

    name: str
    function: MetricFunction


DEFAULT_FIELD_SPECS = (
    FieldSpec("temperature_850", "temperature", 850),
    FieldSpec("geopotential_500", "geopotential", 500),
    FieldSpec("specific_humidity_700", "specific_humidity", 700),
    FieldSpec("sea_surface_temperature", "sea_surface_temperature"),
)


DEFAULT_METRICS = ("rmse", "mae")


def _lead_values(n: int) -> np.ndarray:
    return np.arange(1, int(n) + 1, dtype=int)


def latitude_weights(
    latitude: xr.DataArray | np.ndarray,
    *,
    normalize_like_chaosbench: bool = True,
) -> np.ndarray:
    lat = np.asarray(latitude, dtype=float)
    lat_rad = np.deg2rad(lat) if np.nanmax(np.abs(lat)) > np.pi + 1.0e-6 else lat
    weights = np.cos(lat_rad)
    if normalize_like_chaosbench:
        scale = float(np.mean(weights))
        if scale == 0.0:
            return weights
        weights = weights / scale
    return weights


def weighted_rmse(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    *,
    latitude_name: str = "latitude",
) -> float:
    squared = np.asarray((forecast - truth) ** 2, dtype=float)
    lat_axis = forecast.dims.index(latitude_name)
    weights = latitude_weights(truth[latitude_name])
    reshape = [1] * squared.ndim
    reshape[lat_axis] = weights.shape[0]
    weighted = squared * weights.reshape(reshape)
    return float(np.sqrt(np.mean(weighted)))


def weighted_mae(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    *,
    latitude_name: str = "latitude",
) -> float:
    absolute = np.asarray(np.abs(forecast - truth), dtype=float)
    lat_axis = forecast.dims.index(latitude_name)
    weights = latitude_weights(truth[latitude_name])
    reshape = [1] * absolute.ndim
    reshape[lat_axis] = weights.shape[0]
    weighted = absolute * weights.reshape(reshape)
    return float(np.mean(weighted))


DEFAULT_METRIC_SPECS = {
    "rmse": MetricSpec(
        "rmse",
        lambda forecast, truth: weighted_rmse(forecast, truth),
    ),
    "mae": MetricSpec(
        "mae",
        lambda forecast, truth: weighted_mae(forecast, truth),
    ),
}


def available_metric_names() -> tuple[str, ...]:
    return tuple(DEFAULT_METRICS)


def resolve_metric_specs(metric_names: Sequence[str] | None) -> tuple[MetricSpec, ...]:
    selected = tuple(metric_names) if metric_names is not None else DEFAULT_METRICS
    unknown = sorted(set(selected) - set(DEFAULT_METRIC_SPECS))
    if unknown:
        raise ValueError(f"Unknown metric names: {unknown}")
    return tuple(DEFAULT_METRIC_SPECS[name] for name in selected)


def _metric_column_name(metric_name: str, field_name: str) -> str:
    return f"{metric_name}.{field_name}"


def _select_field(dataset: xr.Dataset, spec: FieldSpec) -> xr.DataArray:
    field = dataset[spec.variable]
    if spec.level is not None:
        field = field.sel({spec.level_name: spec.level}, method="nearest")
    return field


def align_truth_to_forecast(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    lead_dim: str = "lead_day",
    time_name: str = "time",
) -> xr.Dataset:
    aligned = truth
    if time_name in aligned.dims and lead_dim not in aligned.dims:
        aligned = aligned.rename({time_name: lead_dim})
    if lead_dim in aligned.dims and lead_dim in forecast.coords:
        n = int(forecast.sizes[lead_dim])
        aligned = aligned.isel({lead_dim: slice(0, n)}).assign_coords(
            {lead_dim: forecast[lead_dim]}
        )
    return aligned


def build_daily_metric_table(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    field_specs: Sequence[FieldSpec] = DEFAULT_FIELD_SPECS,
    metrics: Sequence[MetricSpec] | None = None,
    lead_dim: str = "lead_day",
    latitude_name: str = "latitude",
) -> dict[str, np.ndarray]:
    truth = align_truth_to_forecast(forecast, truth, lead_dim=lead_dim)
    n_leads = int(forecast.sizes[lead_dim])
    metric_specs = tuple(metrics) if metrics is not None else resolve_metric_specs(None)
    table: dict[str, np.ndarray] = {}
    for metric in metric_specs:
        for spec in field_specs:
            daily = np.empty(n_leads, dtype=float)
            forecast_field = _select_field(forecast, spec)
            truth_field = _select_field(truth, spec)
            for idx in range(n_leads):
                daily[idx] = float(
                    metric.function(
                        forecast_field.isel({lead_dim: idx}),
                        truth_field.isel({lead_dim: idx}),
                    )
                )
            table[_metric_column_name(metric.name, spec.name)] = daily
    return table


def score_forecast_metrics(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    field_specs: Sequence[FieldSpec] = DEFAULT_FIELD_SPECS,
    metrics: Sequence[MetricSpec] | None = None,
    windows: Sequence[LeadTimeWindow] = DEFAULT_S2S_WINDOWS,
    lead_dim: str = "lead_day",
    latitude_name: str = "latitude",
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    daily = build_daily_metric_table(
        forecast,
        truth,
        field_specs=field_specs,
        metrics=metrics,
        lead_dim=lead_dim,
        latitude_name=latitude_name,
    )
    lead_days = (
        forecast[lead_dim].values
        if lead_dim in forecast.coords
        else _lead_values(len(next(iter(daily.values()), ())))
    )
    summary = aggregate_metric_table(daily, lead_days=lead_days, windows=windows)
    return daily, summary


def save_metric_table_csv(
    metric_table: Mapping[str, Sequence[float]],
    path: str | Path,
    *,
    lead_days: Sequence[int] | None,
    lead_name: str = "lead_day",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(next(iter(metric_table.values()), ()))
    lead = _lead_values(n_rows) if lead_days is None else np.asarray(lead_days)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [lead_name, *metric_table.keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_rows):
            row = {lead_name: int(lead[idx])}
            for name, values in metric_table.items():
                row[name] = float(values[idx])
            writer.writerow(row)
    return path


__all__ = [
    "DEFAULT_FIELD_SPECS",
    "DEFAULT_METRICS",
    "DEFAULT_METRIC_SPECS",
    "FieldSpec",
    "MetricSpec",
    "align_truth_to_forecast",
    "available_metric_names",
    "build_daily_metric_table",
    "latitude_weights",
    "resolve_metric_specs",
    "save_metric_table_csv",
    "score_forecast_metrics",
    "weighted_mae",
    "weighted_rmse",
]
