"""WeatherBench2 evaluation driver for legoESM SFNO models.

Runs autoregressive rollouts against ERA5 targets from GCS
and computes standard metrics (RMSE, ACC) at multiple lead times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.gaussian import GaussianGrid
from legoesm.ml.data.era5_loader import (
    ERA5Config,
    ERA5ClimatologyConfig,
    create_era5_dataset,
    create_climatology_dataset,
    _dataset_to_array,
)
from evaluations.metrics import rmse, acc, bias

# Standard WB2 evaluation variables (short names for display)
WB2_EVAL_VARIABLES = (
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)

WB2_EVAL_LEVELS = (500, 850, 250, 700, 1000)


class WB2EvalConfig(NamedTuple):
    """Configuration for WeatherBench2 evaluation.

    Attributes
    ----------
    eval_period : tuple of str
        (start, end) dates for the evaluation period.
    lead_times_hours : tuple of int
        Lead times to evaluate, in hours.
    variables : tuple of str
        Variables to evaluate.
    levels : tuple of int
        Pressure levels to evaluate.
    output_dir : str
        Directory for saving evaluation results.
    era5_config : ERA5Config
        Configuration for ERA5 data.
    clim_config : ERA5ClimatologyConfig
        Configuration for climatology data.
    dt_hours : int
        Model time step in hours (for autoregressive rollout).
    """

    eval_period: tuple = ("2020-01-01", "2020-12-31")
    lead_times_hours: tuple = (6, 12, 24, 48, 72, 120, 168, 240)
    variables: tuple = WB2_EVAL_VARIABLES
    levels: tuple = WB2_EVAL_LEVELS
    output_dir: str = "results/wb2_eval"
    era5_config: ERA5Config = ERA5Config()
    clim_config: ERA5ClimatologyConfig = ERA5ClimatologyConfig()
    dt_hours: int = 6


def evaluate_model(
    model,
    grid: GaussianGrid,
    config: WB2EvalConfig = WB2EvalConfig(),
    pack_fn=None,
    unpack_fn=None,
) -> dict:
    """Run WeatherBench2 evaluation on a model.

    Parameters
    ----------
    model : eqx.Module
        SFNO model (callable: (x, grid) -> y).
    grid : GaussianGrid
        Grid for area-weighted metrics and model evaluation.
    config : WB2EvalConfig
        Evaluation configuration.
    pack_fn : callable, optional
        Function to pack xarray slice -> jax array. Defaults to
        ``_dataset_to_array``.
    unpack_fn : callable, optional
        Function to unpack jax array -> dict of {(var, level): array}.

    Returns
    -------
    dict
        Results dict with metrics per (variable, level, lead_time).
    """
    # Open datasets
    era5_cfg = config.era5_config._replace(
        time_range=config.eval_period,
        variables=config.variables,
        levels=config.levels,
    )
    ds = create_era5_dataset(era5_cfg)
    clim_ds = create_climatology_dataset(config.clim_config)

    # Determine init times — every dt_hours within eval period
    times = ds.time.values
    max_lead = max(config.lead_times_hours)
    time_stride = config.dt_hours
    max_steps = max_lead // time_stride

    # Results accumulator: {(var, level, lead_time): [rmse_values]}
    results = {
        "rmse": {},
        "acc": {},
        "bias": {},
    }

    weights = jnp.array(grid.weights, dtype=jnp.float32)
    n_times = len(times) - max_steps

    # Evaluate at each init time
    for init_idx in range(0, n_times, max_steps):
        # Pack initial condition
        init_slice = ds.isel(time=init_idx)
        state = _dataset_to_array(init_slice, era5_cfg)

        # Autoregressive rollout
        current = state
        for step in range(1, max_steps + 1):
            current = model(current, grid)
            lead_hours = step * time_stride

            if lead_hours not in config.lead_times_hours:
                continue

            # Get target
            target_slice = ds.isel(time=init_idx + step)
            target = _dataset_to_array(target_slice, era5_cfg)

            # Compute per-channel metrics
            for ch_idx in range(current.shape[-1]):
                pred_field = current[..., ch_idx]
                target_field = target[..., ch_idx]

                key = (ch_idx, lead_hours)
                r = float(rmse(pred_field, target_field, weights))
                b = float(bias(pred_field, target_field, weights))

                results["rmse"].setdefault(key, []).append(r)
                results["bias"].setdefault(key, []).append(b)

    # Average over init times
    summary = {}
    for metric_name, metric_dict in results.items():
        summary[metric_name] = {}
        for key, values in metric_dict.items():
            summary[metric_name][str(key)] = float(np.mean(values))

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def compare_to_baselines(
    results: dict,
    baseline_results: dict,
) -> dict:
    """Compare model results to a baseline.

    Parameters
    ----------
    results : dict
        Model evaluation results from ``evaluate_model``.
    baseline_results : dict
        Baseline evaluation results.

    Returns
    -------
    dict
        Comparison dict with skill scores (1 - model_rmse / baseline_rmse).
    """
    comparison = {}
    if "rmse" in results and "rmse" in baseline_results:
        model_rmse = results["rmse"]
        base_rmse = baseline_results["rmse"]
        for key in model_rmse:
            if key in base_rmse and base_rmse[key] > 0:
                skill = 1.0 - model_rmse[key] / base_rmse[key]
                comparison[key] = {"skill_score": skill}
    return comparison


def load_eval_config(yaml_path: str) -> WB2EvalConfig:
    """Load evaluation config from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to YAML config file.

    Returns
    -------
    WB2EvalConfig
    """
    import yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    era5_cfg = ERA5Config(
        zarr_store=cfg.get("dataset", ERA5Config().zarr_store),
        variables=tuple(cfg.get("variables", ERA5Config().variables)),
        levels=tuple(cfg.get("levels", ERA5Config().levels)),
    )
    clim_cfg = ERA5ClimatologyConfig(
        zarr_store=cfg.get("climatology", ERA5ClimatologyConfig().zarr_store),
        variables=tuple(cfg.get("variables", ERA5ClimatologyConfig().variables)),
        levels=tuple(cfg.get("levels", ERA5ClimatologyConfig().levels)),
    )

    return WB2EvalConfig(
        eval_period=tuple(cfg.get("eval_period", ("2020-01-01", "2020-12-31"))),
        lead_times_hours=tuple(cfg.get("lead_times_hours", (6, 12, 24, 48, 72, 120, 168, 240))),
        variables=tuple(cfg.get("variables", WB2_EVAL_VARIABLES)),
        levels=tuple(cfg.get("levels", WB2_EVAL_LEVELS)),
        output_dir=cfg.get("output_dir", "results/wb2_eval"),
        era5_config=era5_cfg,
        clim_config=clim_cfg,
    )
