"""Evaluation helpers for SFNO S2S rollout datasets and campaigns."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from legoesm.ml.s2s.sfno_slab.regrid import build_target_grid


def parse_window_specs(text: str) -> dict[str, tuple[int, int]]:
    """Parse a comma-separated window specification like ``wk3_4=15:28,wk5_6=29:42``."""
    windows: dict[str, tuple[int, int]] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, span = chunk.split("=", 1)
        start, end = span.split(":", 1)
        windows[name.strip()] = (int(start), int(end))
    return windows


def aggregate_long_records(
    records: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str],
    value_key: str = "value",
) -> list[dict[str, object]]:
    """Average long-format records over dimensions not listed in ``group_keys``."""
    grouped: dict[tuple[object, ...], list[float]] = {}
    for record in records:
        key = tuple(record[name] for name in group_keys)
        grouped.setdefault(key, []).append(float(record[value_key]))

    aggregated: list[dict[str, object]] = []
    for key in sorted(grouped):
        row = {name: value for name, value in zip(group_keys, key, strict=True)}
        row[value_key] = float(np.nanmean(np.asarray(grouped[key], dtype=float)))
        aggregated.append(row)
    return aggregated


def select_available_fields(ds: xr.Dataset, requested: list[str] | None = None) -> list[str]:
    """Return requested rollout channels that are present in the dataset."""
    available: list[str] = []
    if "channel" in ds.coords:
        available.extend(str(value) for value in ds["channel"].values.tolist())
    if "forcing_channel" in ds.coords:
        available.extend(str(value) for value in ds["forcing_channel"].values.tolist())
    if requested is None:
        return available
    return [field for field in requested if field in available]


def _field_index_maps(ds: xr.Dataset) -> tuple[dict[str, int], dict[str, int]]:
    channel_index = (
        {str(label): idx for idx, label in enumerate(ds["channel"].values.tolist())}
        if "channel" in ds.coords else {}
    )
    forcing_index = (
        {str(label): idx for idx, label in enumerate(ds["forcing_channel"].values.tolist())}
        if "forcing_channel" in ds.coords else {}
    )
    return channel_index, forcing_index


def _extract_field_pair(ds: xr.Dataset, field: str) -> tuple[np.ndarray, np.ndarray]:
    channel_index, forcing_index = _field_index_maps(ds)
    prediction = np.asarray(ds["prediction"].values, dtype=np.float64)
    target = np.asarray(ds["target"].values, dtype=np.float64)
    forcing = (
        np.asarray(ds["forcing"].values, dtype=np.float64)
        if "forcing" in ds.data_vars and "target_forcing" in ds.data_vars else None
    )
    target_forcing = (
        np.asarray(ds["target_forcing"].values, dtype=np.float64)
        if "target_forcing" in ds.data_vars else None
    )

    if field in channel_index:
        idx = channel_index[field]
        return prediction[..., idx], target[..., idx]
    if forcing is not None and target_forcing is not None and field in forcing_index:
        idx = forcing_index[field]
        return forcing[..., idx], target_forcing[..., idx]
    raise KeyError(f"Field {field!r} is not available for evaluation.")


@jax.jit
def _weighted_rmse_series(prediction: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    def per_lead(pred_slice: jnp.ndarray, target_slice: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.mean(((pred_slice - target_slice) ** 2) * weights[:, None]))

    return jax.vmap(per_lead)(prediction, target)


@jax.jit
def _weighted_mae_series(prediction: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    def per_lead(pred_slice: jnp.ndarray, target_slice: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.abs(pred_slice - target_slice) * weights[:, None])

    return jax.vmap(per_lead)(prediction, target)


@jax.jit
def _weighted_crps_series(ensemble: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    def per_lead(member_slice: jnp.ndarray, target_slice: jnp.ndarray) -> jnp.ndarray:
        obs_term = jnp.mean(jnp.abs(member_slice - target_slice[None, ...]), axis=0)
        pairwise = jnp.abs(member_slice[:, None, ...] - member_slice[None, :, ...])
        crps_field = obs_term - 0.5 * jnp.mean(pairwise, axis=(0, 1))
        # Same n_lat/sum(w) resolution-independence correction the losses in
        # ml/loss.py apply (#1413). Without it this rollout metric sits on a
        # different scale from the AFCRPS reported by validate_s2s_step, so the
        # two could not be compared — and the gap grows with resolution.
        n_lat = weights.shape[0]
        return (jnp.mean(crps_field * weights[:, None])
                * n_lat / jnp.sum(weights))

    return jax.vmap(per_lead, in_axes=(1, 0))(ensemble, target)


def compute_daily_metrics(ds: xr.Dataset, *, fields: list[str] | None = None) -> list[dict[str, float | str | int]]:
    """Compute area-weighted daily RMSE and MAE for the selected rollout fields."""
    if "gaussian_n_max" not in ds.attrs:
        raise KeyError("Rollout dataset is missing gaussian_n_max in attrs.")
    fields = select_available_fields(ds, requested=fields)
    grid = build_target_grid(int(ds.attrs["gaussian_n_max"])).grid
    weights = np.asarray(grid.weights, dtype=np.float64)
    normalized_weights = weights / np.mean(weights)
    lead_days = np.asarray(ds["lead_day"].values, dtype=np.int32)
    rows: list[dict[str, float | str | int]] = []
    for field in fields:
        field_prediction, field_target = _extract_field_pair(ds, field)
        diff = field_prediction - field_target
        weighted_mse = np.mean((diff**2) * normalized_weights[None, :, None], axis=(1, 2))
        weighted_mae = np.mean(np.abs(diff) * normalized_weights[None, :, None], axis=(1, 2))
        for lead_day, rmse, mae in zip(lead_days, np.sqrt(weighted_mse), weighted_mae, strict=True):
            rows.append(
                {
                    "field": field,
                    "lead_day": int(lead_day),
                    "rmse": float(rmse),
                    "mae": float(mae),
                }
            )
    return rows


def compute_ensemble_daily_metrics(
    member_datasets: list[xr.Dataset],
    *,
    fields: list[str] | None = None,
) -> list[dict[str, float | str | int]]:
    """Compute daily ensemble-mean RMSE/MAE and ensemble CRPS for selected fields."""
    if not member_datasets:
        raise ValueError("At least one member rollout is required for ensemble metrics.")
    reference = member_datasets[0]
    if "gaussian_n_max" not in reference.attrs:
        raise KeyError("Rollout dataset is missing gaussian_n_max in attrs.")

    fields = select_available_fields(reference, requested=fields)
    grid = build_target_grid(int(reference.attrs["gaussian_n_max"])).grid
    weights = jnp.asarray(np.asarray(grid.weights, dtype=np.float32) / np.mean(np.asarray(grid.weights, dtype=np.float32)))
    lead_days = np.asarray(reference["lead_day"].values, dtype=np.int32)

    rows: list[dict[str, float | str | int]] = []
    for field in fields:
        member_prediction = []
        target_array = None
        for ds in member_datasets:
            pred, truth = _extract_field_pair(ds, field)
            member_prediction.append(pred.astype(np.float32, copy=False))
            if target_array is None:
                target_array = truth.astype(np.float32, copy=False)
        if target_array is None:
            raise RuntimeError(f"Unable to build ensemble metric stack for field {field!r}.")

        ensemble = jnp.asarray(np.stack(member_prediction, axis=0))
        target_jax = jnp.asarray(target_array)
        ensemble_mean = jnp.mean(ensemble, axis=0)
        rmse = np.asarray(_weighted_rmse_series(ensemble_mean, target_jax, weights), dtype=np.float64)
        mae = np.asarray(_weighted_mae_series(ensemble_mean, target_jax, weights), dtype=np.float64)
        crps = np.asarray(_weighted_crps_series(ensemble, target_jax, weights), dtype=np.float64)

        for lead_day, rmse_value, mae_value, crps_value in zip(lead_days, rmse, mae, crps, strict=True):
            rows.append(
                {
                    "field": field,
                    "lead_day": int(lead_day),
                    "rmse": float(rmse_value),
                    "mae": float(mae_value),
                    "crps": float(crps_value),
                    "n_members": int(ensemble.shape[0]),
                }
            )
    return rows


def summarize_window_metrics(
    daily_rows: list[dict[str, float | str | int]],
    *,
    windows: dict[str, tuple[int, int]],
) -> list[dict[str, float | str]]:
    """Average daily metrics into named lead windows."""
    rows: list[dict[str, float | str]] = []
    by_field: dict[str, list[dict[str, float | str | int]]] = {}
    for row in daily_rows:
        by_field.setdefault(str(row["field"]), []).append(row)

    for field, field_rows in by_field.items():
        for window_name, (start_day, end_day) in windows.items():
            selected = [
                row for row in field_rows
                if start_day <= int(row["lead_day"]) <= end_day
            ]
            if not selected:
                continue
            summary: dict[str, float | str] = {"field": field, "window": window_name}
            numeric_keys = [
                key for key in selected[0].keys()
                if key not in {"field", "lead_day"}
            ]
            for key in numeric_keys:
                summary[key] = float(np.mean([float(row[key]) for row in selected]))
            rows.append(summary)
    return rows


def write_metric_rows(rows: list[dict[str, float | str | int]], path: str | Path) -> None:
    """Write metric rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "aggregate_long_records",
    "compute_ensemble_daily_metrics",
    "compute_daily_metrics",
    "parse_window_specs",
    "select_available_fields",
    "summarize_window_metrics",
    "write_metric_rows",
]
