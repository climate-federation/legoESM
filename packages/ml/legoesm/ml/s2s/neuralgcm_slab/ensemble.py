"""Ensemble utilities for NeuralGCM slab-ocean experiments."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.ml.s2s.neuralgcm_slab.evaluation import (
    DEFAULT_S2S_WINDOWS,
    LeadTimeWindow,
    aggregate_metric_table,
)
from legoesm.ml.s2s.neuralgcm_slab.metrics import (
    align_truth_to_forecast,
    latitude_weights,
    weighted_mae,
    weighted_rmse,
)


@dataclass(frozen=True)
class HeadlineField:
    """Headline variable definition used in ensemble metrics and plots."""

    name: str
    variable: str
    level: int | None = None
    scale: float = 1.0
    colorbar_label: str = ""
    title: str = ""


HEADLINE_FIELDS = (
    HeadlineField("t850", "temperature", 850, 1.0, "K", "T850"),
    HeadlineField("z500", "geopotential", 500, 1.0 / constants.g, "m", "Z500"),
    HeadlineField("q700", "specific_humidity", 700, 1.0, "kg kg-1", "Q700"),
    HeadlineField("sst", "sea_surface_temperature", None, 1.0, "K", "SST"),
)


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for pseudo-ensemble initial-condition perturbations."""

    base_seed: int = 20220101
    perturbation_kind: str = "correlated_gaussian"
    latitude_correlation_degrees: float = 10.0
    longitude_correlation_degrees: float = 16.0
    kernel_truncation_sigma: float = 4.0
    temperature_amplitude: float = 5.0e-3
    specific_humidity_amplitude: float = 5.0e-6
    u_component_of_wind_amplitude: float = 2.0e-3
    v_component_of_wind_amplitude: float = 2.0e-3
    sea_surface_temperature_amplitude: float = 1.0e-2


PERTURBATION_VARIABLES = (
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "sea_surface_temperature",
)
PERTURBATION_GROUPS = {
    "temperature": "thermodynamic",
    "specific_humidity": "thermodynamic",
    "sea_surface_temperature": "thermodynamic",
    "u_component_of_wind": "wind",
    "v_component_of_wind": "wind",
}


MetricFunction = Callable[[xr.DataArray, xr.DataArray], float]


def member_name(member_index: int) -> str:
    return f"seed_{member_index:02d}"


def member_seed(
    member_index: int,
    *,
    config: PerturbationConfig = PerturbationConfig(),
) -> int:
    return config.base_seed + int(member_index)


def _open_dataset(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _select_headline_field(dataset: xr.Dataset, field: HeadlineField) -> xr.DataArray:
    data = dataset[field.variable]
    if field.level is not None:
        data = data.sel(level=field.level, method="nearest")
    return data * field.scale


def _weighted_spatial_mean(
    field: xr.DataArray,
    *,
    latitude_name: str = "latitude",
) -> float:
    array = np.asarray(field, dtype=float)
    weights = latitude_weights(field[latitude_name])
    lat_axis = field.dims.index(latitude_name)
    reshape = [1] * array.ndim
    reshape[lat_axis] = weights.shape[0]
    weighted = array * weights.reshape(reshape)
    return float(np.mean(weighted))


def _grid_spacing(values: np.ndarray, *, periodic: bool) -> float:
    coords = np.asarray(values, dtype=float)
    diffs = np.diff(coords)
    if periodic and coords.size > 1:
        wrap = (coords[0] + 360.0) - coords[-1]
        positive = np.concatenate([np.abs(diffs), np.asarray([abs(wrap)], dtype=float)])
    else:
        positive = np.abs(diffs)
    return float(np.mean(positive)) if positive.size else 1.0


def _gaussian_kernel1d(sigma_grid: float, *, truncate: float) -> np.ndarray:
    radius = max(int(np.ceil(float(truncate) * float(sigma_grid))), 1)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / float(sigma_grid)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _convolve_wrap(field: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(field, pad_width, mode="wrap")
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, padded)


def _convolve_reflect(field: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(field, pad_width, mode="reflect")
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis, padded)


def generate_correlated_gaussian_noise(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    seed: int,
    *,
    config: PerturbationConfig,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    field = rng.standard_normal((latitude.size, longitude.size), dtype=np.float32)
    lat_spacing = _grid_spacing(np.asarray(latitude), periodic=False)
    lon_spacing = _grid_spacing(np.asarray(longitude), periodic=True)
    lat_sigma = max(config.latitude_correlation_degrees / max(lat_spacing, 1.0e-6), 1.0)
    lon_sigma = max(config.longitude_correlation_degrees / max(lon_spacing, 1.0e-6), 1.0)
    field = _convolve_reflect(
        field,
        _gaussian_kernel1d(lat_sigma, truncate=config.kernel_truncation_sigma),
        axis=0,
    )
    field = _convolve_wrap(
        field,
        _gaussian_kernel1d(lon_sigma, truncate=config.kernel_truncation_sigma),
        axis=1,
    )
    std = float(field.std())
    if std > 0.0:
        field /= std
    return field


def _broadcast_noise_to_field(field: xr.DataArray, noise: np.ndarray) -> np.ndarray:
    noise_field = xr.DataArray(
        noise,
        coords={"latitude": field["latitude"], "longitude": field["longitude"]},
        dims=("latitude", "longitude"),
    )
    spatial_dims = tuple(dim for dim in field.dims if dim in {"latitude", "longitude"})
    noise_field = noise_field.transpose(*spatial_dims)
    broadcast, _ = xr.broadcast(noise_field, field)
    broadcast = broadcast.transpose(*field.dims)
    return np.asarray(broadcast, dtype=np.float32)


def _vertical_taper(field: xr.DataArray) -> np.ndarray:
    if "level" not in field.dims:
        return np.ones(field.shape, dtype=np.float32)
    levels = np.asarray(field["level"], dtype=float)
    max_level = float(np.max(levels)) if levels.size else 1.0
    normalized = np.clip(levels / max(max_level, 1.0), 0.0, 1.0)
    taper = normalized[:, None, None].astype(np.float32)
    return np.broadcast_to(taper, field.shape)


def apply_initial_perturbations(
    initial_dataset: xr.Dataset,
    *,
    member_index: int,
    config: PerturbationConfig = PerturbationConfig(),
) -> tuple[xr.Dataset, dict[str, Any]]:
    perturbed = initial_dataset.copy(deep=True)
    base = member_seed(member_index, config=config)
    metadata: dict[str, Any] = {
        "member_index": int(member_index),
        "seed": int(base),
        "perturbation_kind": config.perturbation_kind,
        "variables": {},
    }
    amplitudes = {
        "temperature": config.temperature_amplitude,
        "specific_humidity": config.specific_humidity_amplitude,
        "u_component_of_wind": config.u_component_of_wind_amplitude,
        "v_component_of_wind": config.v_component_of_wind_amplitude,
        "sea_surface_temperature": config.sea_surface_temperature_amplitude,
    }
    latitude = perturbed["latitude"]
    longitude = perturbed["longitude"]
    group_seeds = {
        "thermodynamic": base + 101,
        "wind": base + 202,
    }
    group_noise = {
        group: generate_correlated_gaussian_noise(latitude, longitude, seed, config=config)
        for group, seed in group_seeds.items()
    }

    for variable in PERTURBATION_VARIABLES:
        if variable not in perturbed:
            continue
        group = PERTURBATION_GROUPS[variable]
        seed = group_seeds[group]
        noise = group_noise[group]
        field = perturbed[variable]
        delta = (
            amplitudes[variable]
            * _broadcast_noise_to_field(field, noise)
            * _vertical_taper(field)
        )
        updated = field + delta
        if variable == "specific_humidity":
            updated = updated.clip(min=0.0)
        perturbed[variable] = updated

        delta_np = np.asarray(delta, dtype=float)
        metadata["variables"][variable] = {
            "seed": int(seed),
            "group": group,
            "amplitude": float(amplitudes[variable]),
            "latitude_correlation_degrees": float(config.latitude_correlation_degrees),
            "longitude_correlation_degrees": float(config.longitude_correlation_degrees),
            "delta_mean": float(delta_np.mean()),
            "delta_std": float(delta_np.std()),
            "delta_min": float(delta_np.min()),
            "delta_max": float(delta_np.max()),
        }

    return perturbed, metadata


def _build_member_daily_metric_table(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField],
    metric_function: MetricFunction,
    lead_dim: str,
) -> dict[str, np.ndarray]:
    truth = align_truth_to_forecast(forecast, truth, lead_dim=lead_dim)
    n_leads = int(forecast.sizes[lead_dim])
    table: dict[str, np.ndarray] = {}
    for field in fields:
        forecast_field = _select_headline_field(forecast, field)
        truth_field = _select_headline_field(truth, field)
        values = np.empty(n_leads, dtype=float)
        for idx in range(n_leads):
            values[idx] = metric_function(
                forecast_field.isel({lead_dim: idx}),
                truth_field.isel({lead_dim: idx}),
            )
        table[field.name] = values
    return table


def build_member_daily_rmse_table(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField] = HEADLINE_FIELDS,
    lead_dim: str = "lead_day",
) -> dict[str, np.ndarray]:
    return _build_member_daily_metric_table(
        forecast,
        truth,
        fields=fields,
        metric_function=weighted_rmse,
        lead_dim=lead_dim,
    )


def build_member_daily_mae_table(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField] = HEADLINE_FIELDS,
    lead_dim: str = "lead_day",
) -> dict[str, np.ndarray]:
    return _build_member_daily_metric_table(
        forecast,
        truth,
        fields=fields,
        metric_function=weighted_mae,
        lead_dim=lead_dim,
    )


def _build_ensemble_mean_daily_metric_table(
    forecasts: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField],
    metric_function: MetricFunction,
    lead_dim: str,
) -> dict[str, np.ndarray]:
    if not forecasts:
        raise ValueError("At least one forecast is required to compute ensemble-mean metrics")
    reference = forecasts[0]
    truth = align_truth_to_forecast(reference, truth, lead_dim=lead_dim)
    n_leads = int(reference.sizes[lead_dim])
    table: dict[str, np.ndarray] = {}
    for field in fields:
        member_fields = [_select_headline_field(forecast, field) for forecast in forecasts]
        ensemble_mean = xr.concat(member_fields, dim="member").mean("member")
        truth_field = _select_headline_field(truth, field)
        values = np.empty(n_leads, dtype=float)
        for idx in range(n_leads):
            values[idx] = metric_function(
                ensemble_mean.isel({lead_dim: idx}),
                truth_field.isel({lead_dim: idx}),
            )
        table[field.name] = values
    return table


def build_ensemble_mean_daily_rmse_table(
    forecasts: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField] = HEADLINE_FIELDS,
    lead_dim: str = "lead_day",
) -> dict[str, np.ndarray]:
    return _build_ensemble_mean_daily_metric_table(
        forecasts,
        truth,
        fields=fields,
        metric_function=weighted_rmse,
        lead_dim=lead_dim,
    )


def build_ensemble_mean_daily_mae_table(
    forecasts: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField] = HEADLINE_FIELDS,
    lead_dim: str = "lead_day",
) -> dict[str, np.ndarray]:
    return _build_ensemble_mean_daily_metric_table(
        forecasts,
        truth,
        fields=fields,
        metric_function=weighted_mae,
        lead_dim=lead_dim,
    )


def _empirical_crps(ensemble: np.ndarray, truth: np.ndarray) -> np.ndarray:
    term_obs = np.mean(np.abs(ensemble - truth[None, ...]), axis=0)
    pairwise = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
    term_pairwise = 0.5 * np.mean(pairwise, axis=(0, 1))
    return term_obs - term_pairwise


def _align_slice_to_reference(field: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    aligned = field
    for dim in reference.dims:
        if dim not in aligned.dims:
            continue
        ref_coord = reference[dim]
        if dim in aligned.coords:
            if int(aligned.sizes[dim]) == int(ref_coord.size):
                aligned = aligned.assign_coords({dim: ref_coord})
            else:
                aligned = aligned.sel({dim: ref_coord}, method="nearest")
    return aligned.transpose(*reference.dims)


def build_ensemble_daily_crps_table(
    forecasts: Sequence[xr.Dataset],
    truth: xr.Dataset,
    *,
    fields: Sequence[HeadlineField] = HEADLINE_FIELDS,
    lead_dim: str = "lead_day",
) -> dict[str, np.ndarray]:
    if not forecasts:
        raise ValueError("At least one forecast is required to compute ensemble CRPS")
    reference = forecasts[0]
    truth = align_truth_to_forecast(reference, truth, lead_dim=lead_dim)
    n_leads = int(reference.sizes[lead_dim])
    table: dict[str, np.ndarray] = {}
    for field in fields:
        truth_field = _select_headline_field(truth, field)
        member_fields = [_select_headline_field(forecast, field) for forecast in forecasts]
        values = np.empty(n_leads, dtype=float)
        for idx in range(n_leads):
            reference_slice = member_fields[0].isel({lead_dim: idx})
            truth_slice = _align_slice_to_reference(
                truth_field.isel({lead_dim: idx}),
                reference_slice,
            )
            ensemble = np.stack(
                [
                    np.asarray(
                        _align_slice_to_reference(member.isel({lead_dim: idx}), reference_slice),
                        dtype=float,
                    )
                    for member in member_fields
                ],
                axis=0,
            )
            crps_field = xr.DataArray(
                _empirical_crps(ensemble, np.asarray(truth_slice, dtype=float)),
                coords=reference_slice.coords,
                dims=reference_slice.dims,
            )
            values[idx] = _weighted_spatial_mean(crps_field)
        table[field.name] = values
    return table


def aggregate_daily_series(
    metric_table: Mapping[str, Sequence[float]],
    *,
    lead_days: Sequence[int],
    windows: Sequence[LeadTimeWindow] = DEFAULT_S2S_WINDOWS,
) -> dict[str, dict[str, float]]:
    return aggregate_metric_table(metric_table, lead_days=lead_days, windows=windows)


def save_long_records_csv(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names = list(records[0].keys()) if records else []
    else:
        names = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


def open_member_forecasts(
    case_dir: str | Path,
    *,
    member_indices: Sequence[int],
    experiment: str,
) -> list[xr.Dataset]:
    case_dir = Path(case_dir)
    forecasts: list[xr.Dataset] = []
    for member_index in member_indices:
        path = case_dir / member_name(member_index) / f"{experiment}.nc"
        forecasts.append(_open_dataset(path))
    return forecasts


__all__ = [
    "HEADLINE_FIELDS",
    "HeadlineField",
    "PERTURBATION_GROUPS",
    "PERTURBATION_VARIABLES",
    "PerturbationConfig",
    "aggregate_daily_series",
    "apply_initial_perturbations",
    "build_ensemble_daily_crps_table",
    "build_ensemble_mean_daily_mae_table",
    "build_ensemble_mean_daily_rmse_table",
    "build_member_daily_mae_table",
    "build_member_daily_rmse_table",
    "generate_correlated_gaussian_noise",
    "member_name",
    "member_seed",
    "open_member_forecasts",
    "save_long_records_csv",
]
