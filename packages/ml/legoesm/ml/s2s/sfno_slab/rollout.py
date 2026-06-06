"""Checkpoint rollout utilities for the local SFNO S2S workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import xarray as xr

from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.s2s.sfno_slab.data import (
    atmospheric_param_labels,
    available_s2s_dates,
    denormalize_atmospheric_channels,
    denormalize_forcing_channels,
    forcing_channel_labels,
    load_normalization_bundle,
    load_s2s_sample,
    resolve_s2s_sample_dates,
)
from legoesm.ml.s2s.sfno_slab.regrid import build_target_grid
from legoesm.ml.s2s.sfno_slab.training import (
    S2SStochasticConfig,
    cast_model_to_float32,
    rollout_with_forcing,
)
from legoesm.ml.training import load_checkpoint


def metadata_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the default metadata path associated with a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent / "metadata.json"


def save_training_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Save training metadata as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_training_metadata(path: str | Path) -> dict[str, Any]:
    """Load training metadata from JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sfno_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    gaussian_n_max: int,
    in_channels: int,
    out_channels: int,
    embed_dim: int,
    n_blocks: int,
    mlp_expansion: int,
    residual_prediction: bool,
) -> tuple[SFNO, Any]:
    """Rebuild an SFNO and load its weights from a checkpoint."""
    target_grid = build_target_grid(gaussian_n_max)
    model_template = SFNO(
        SFNOConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            mlp_expansion=mlp_expansion,
            residual_prediction=residual_prediction,
        ),
        grid=target_grid.grid,
        key=jax.random.PRNGKey(0),
    )
    model_template = cast_model_to_float32(model_template)
    model = load_checkpoint(model_template, checkpoint_path)
    model = cast_model_to_float32(model)
    return model, target_grid


def sample_index_from_date(
    available_dates: list[str],
    config,
    sample_date: str,
) -> int:
    """Resolve a YYYYMMDD date into the corresponding S2S sample index."""
    if sample_date not in available_dates:
        raise KeyError(f"Date {sample_date!r} is not available in the requested dataset window.")
    candidate = available_dates.index(sample_date)
    max_index = len(available_dates) - config.lead_time - config.n_steps + 1
    if candidate >= max_index:
        raise IndexError(
            f"Date {sample_date} is too close to the end of the record for "
            f"lead_time={config.lead_time}, n_steps={config.n_steps}."
        )
    return candidate


def rollout_to_dataset(
    model: eqx.Module,
    data_config,
    *,
    sample_index: int,
    stochastic_seed: int = 0,
    stochastic_config: S2SStochasticConfig = S2SStochasticConfig(),
    tendency_prediction: bool = False,
) -> xr.Dataset:
    """Run one denormalized rollout and package it as an xarray Dataset.

    This is the uncoupled control path. The saved SST field is simply the
    forcing sequence presented to the atmosphere model, which makes it a useful
    baseline against the slab-coupled rollout written by
    :func:`coupled_rollout_to_dataset`.
    """
    target_grid = build_target_grid(data_config.gaussian_n_max)
    norms = load_normalization_bundle(data_config)
    input_date, target_dates = resolve_s2s_sample_dates(data_config, sample_index=sample_index)
    initial_input, target, forcing = load_s2s_sample(
        data_config,
        sample_index=sample_index,
        target_grid=target_grid,
        norms=norms,
    )

    prediction = np.asarray(
        rollout_with_forcing(
            model,
            jax.numpy.asarray(initial_input),
            jax.numpy.asarray(forcing),
            target_grid.grid,
            rng_key=jax.random.PRNGKey(int(stochastic_seed)),
            stochastic_config=stochastic_config,
            tendency_prediction=tendency_prediction,
        )
    )

    n_atmos_channels = len(atmospheric_param_labels(data_config))
    initial_atmos = initial_input[..., :n_atmos_channels]
    initial_forcing = initial_input[..., n_atmos_channels:]

    prediction_physical = denormalize_atmospheric_channels(prediction, data_config, norms=norms)
    target_physical = denormalize_atmospheric_channels(target, data_config, norms=norms)
    forcing_physical = denormalize_forcing_channels(forcing, data_config, norms=norms)
    initial_atmos_physical = denormalize_atmospheric_channels(initial_atmos, data_config, norms=norms)
    initial_forcing_physical = denormalize_forcing_channels(initial_forcing, data_config, norms=norms)

    channel_labels = atmospheric_param_labels(data_config)
    forcing_labels = forcing_channel_labels(data_config)
    lead_days = np.arange(1, prediction.shape[0] + 1, dtype=np.int32)

    ds = xr.Dataset(
        data_vars={
            "prediction": (
                ("lead_day", "latitude", "longitude", "channel"),
                prediction_physical.astype(np.float32, copy=False),
            ),
            "target": (
                ("lead_day", "latitude", "longitude", "channel"),
                target_physical.astype(np.float32, copy=False),
            ),
            "forcing": (
                ("lead_day", "latitude", "longitude", "forcing_channel"),
                forcing_physical.astype(np.float32, copy=False),
            ),
            "target_forcing": (
                ("lead_day", "latitude", "longitude", "forcing_channel"),
                forcing_physical.astype(np.float32, copy=False),
            ),
            "initial_atmosphere": (
                ("latitude", "longitude", "channel"),
                initial_atmos_physical.astype(np.float32, copy=False),
            ),
            "initial_forcing": (
                ("latitude", "longitude", "forcing_channel"),
                initial_forcing_physical.astype(np.float32, copy=False),
            ),
        },
        coords={
            "lead_day": lead_days,
            "target_date": ("lead_day", np.asarray(target_dates, dtype="U8")),
            "latitude": target_grid.latitude_deg.astype(np.float32, copy=False),
            "longitude": target_grid.longitude_deg.astype(np.float32, copy=False),
            "channel": np.asarray(channel_labels, dtype="U16"),
            "forcing_channel": np.asarray(forcing_labels, dtype="U16"),
        },
        attrs={
            "init_date": input_date,
            "sample_index": int(sample_index),
            "gaussian_n_max": int(data_config.gaussian_n_max),
            "normalize": int(bool(data_config.normalize)),
            "coupled": 0,
            "stochastic_seed": int(stochastic_seed),
            "tendency_prediction": int(bool(tendency_prediction)),
        },
    )

    if "sosstsst" in forcing_labels:
        sst_index = forcing_labels.index("sosstsst")
        # In the uncoupled control the SST diagnostic is just the prescribed
        # forcing sequence that the atmosphere model received.
        ds["sea_surface_temperature"] = (
            ("lead_day", "latitude", "longitude"),
            forcing_physical[..., sst_index].astype(np.float32, copy=False),
        )
        ds["target_sea_surface_temperature"] = (
            ("lead_day", "latitude", "longitude"),
            forcing_physical[..., sst_index].astype(np.float32, copy=False),
        )
    return ds


__all__ = [
    "build_sfno_from_checkpoint",
    "load_training_metadata",
    "metadata_path_for_checkpoint",
    "rollout_to_dataset",
    "sample_index_from_date",
    "save_training_metadata",
]
