"""The sigma-lane lid is a recorded config field (GridConfig.sigma_top), reaches
create_sigma_coordinate, and the hybrid-only fields are refused on the sigma
lane instead of being silently inert (the resolved config used to claim a
2 hPa lid while the model ran 10 hPa)."""
from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from legoesm.driver.config import ExperimentConfig, GridConfig
from legoesm.grids.vertical import create_sigma_coordinate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "scripts", "run"))
from run_amip import build_arg_parser  # noqa: E402


def test_sigma_top_default_and_lid():
    assert GridConfig().sigma_top == 0.01
    for top in (0.01, 0.002):
        c = create_sigma_coordinate(30, sigma_top=top, dtype=jnp.float64)
        assert float(c.sigma_half[0]) == top


def test_sigma_lane_refuses_inert_hybrid_fields_and_bad_lid():
    ExperimentConfig(grid=GridConfig(vertical_coord="sigma")).validate_strict()
    with pytest.raises(ValueError, match="sigma_top"):
        ExperimentConfig(grid=GridConfig(vertical_coord="sigma", p_top_Pa=100.0)).validate_strict()
    with pytest.raises(ValueError, match="sigma_top"):
        ExperimentConfig(grid=GridConfig(vertical_coord="sigma", sigma_top=1.5)).validate_strict()
    with pytest.raises(ValueError, match="sigma_top"):
        ExperimentConfig(grid=GridConfig(vertical_coord="hybrid", sigma_top=0.002)).validate_strict()


def test_run_amip_sigma_top_roundtrip():
    args = build_arg_parser().parse_args(["--sigma-top", "0.002"])
    assert args.sigma_top == 0.002
    assert build_arg_parser().parse_args([]).sigma_top is None
