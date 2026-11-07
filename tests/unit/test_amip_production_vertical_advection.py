"""Van Leer vertical advection is gated to the sigma lane by validate_strict.

It was the production default from the 2026-09-16 user decision until
2026-09-23, when the CAM6 suite became production; it now lives in
config/amip/amip_sundqvist_l36.yaml, which is the deck this pins.  The
production deck's own (hybrid-lane) choice is pinned in
tests/unit/test_mpas_div_damp4.py."""
from __future__ import annotations

import pathlib

import pytest
import yaml

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "config" / "amip" / "amip_sundqvist_l36.yaml"


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
