"""Van Leer vertical advection is the production default (2026-09-16 user
decision, see the deck comment) and is gated to the sigma lane by
validate_strict."""
from __future__ import annotations

import pathlib

import pytest
import yaml

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "config" / "amip" / "amip_production.yaml"


def test_deck_uses_van_leer_on_the_sigma_lane():
    doc = yaml.safe_load(_CONFIG.read_text())
    assert doc["mpas_vert_advection_scheme"] == "van_leer"
    assert doc["vertical_coord"] == "sigma"


def test_validate_strict_accepts_sigma_and_refuses_hybrid():
    ExperimentConfig(
        dycore=DycoreConfig(discretization="mpas", mpas_vert_advection_scheme="van_leer"),
        grid=GridConfig(grid_type="mpas", vertical_coord="sigma", nlev=30, resolution=6),
    ).validate_strict()
    with pytest.raises(ValueError, match="mpas_vert_advection_scheme"):
        ExperimentConfig(
            dycore=DycoreConfig(discretization="mpas", mpas_vert_advection_scheme="van_leer"),
            grid=GridConfig(grid_type="mpas", vertical_coord="hybrid", nlev=30, resolution=6),
        ).validate_strict()
