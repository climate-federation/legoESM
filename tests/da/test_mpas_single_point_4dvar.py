"""Tests for the MPAS single-point T500 experiment configuration and helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run.mpas_4dvar_single.common import load_config, nearest_cell

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "4DVar_single" / "experiment.yaml"


def test_experiment_configuration_matches_documented_observation() -> None:
    config = load_config(CONFIG_PATH)
    observation = config["observation"]
    assert observation["variable"] == "T500"
    assert observation["latitude_degrees_north"] == 45.0
    assert observation["longitude_degrees_east"] == 310.0
    assert observation["increment_k"] == 2.0
    assert observation["error_standard_deviation_k"] == 1.0
    assert observation["four_dvar_time_hours"] == 6.0
    assert config["diagnostics"]["forecast_hours"] == [0.0, 2.0, 4.0, 6.0]


def test_nearest_cell_handles_west_longitude_as_east_degrees() -> None:
    mesh = SimpleNamespace(
        latCell=np.deg2rad(np.asarray([45.0, 45.0, 10.0])),
        lonCell=np.deg2rad(np.asarray([310.0, 300.0, 310.0])),
    )
    assert nearest_cell(mesh, 45.0, -50.0) == 0


def test_minimizer_status_names_line_search_failure() -> None:
    import jax.numpy as jnp

    from legoesm.da.minimizer import minimize_lbfgs
    from scripts.run.mpas_4dvar_single.run_assimilation import (
        minimizer_status_message,
    )

    x0 = jnp.array([1.0, -2.0])
    bowl = minimize_lbfgs(lambda x: (jnp.sum(x**2), 2.0 * x), x0, gtol=1e-8)
    wrong_sign = minimize_lbfgs(lambda x: (jnp.sum(x**2), -2.0 * x), x0)
    assert minimizer_status_message(bowl) == "converged"
    assert bool(wrong_sign.line_search_failed)
    assert minimizer_status_message(wrong_sign) == "line search failed"
