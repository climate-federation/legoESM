"""Configuration tests for the MPAS temperature single-observation 3DVar."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "3DVar_single" / "experiment.yaml"


def test_configuration_matches_finalized_experiment() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    observation = config["observation"]
    covariance = config["background_error"]
    assert observation["variable"] == "temperature"
    assert observation["latitude_degrees_north"] == 40.4113
    assert observation["longitude_degrees_east"] == -38.68
    assert observation["pressure_hpa"] == 800.0
    assert observation["increment_k"] == 1.0
    assert observation["error_standard_deviation_k"] == 1.0
    assert covariance["horizontal_length_scale_multiplier"] == 1.0
    assert covariance["standard_deviation_multiplier"] == 3.0
    assert config["diagnostics"]["pressure_levels_hpa"] == [675.0, 800.0, 925.0]
