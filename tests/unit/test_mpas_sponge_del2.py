"""CAM-style top diffusion sponge for the MPAS dycore: a per-level multiplier
on the del2 viscosity, geometric from `factor` at the model top to 1 below the
top n layers; off state (0 / 1.0) is exactly ones so production is byte-identical."""
from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig, sponge_del2_profile)
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "scripts", "run"))
from run_amip import build_arg_parser, build_config_from_args  # noqa: E402


def test_profile_off_state_is_exact_ones_and_ramp_is_geometric():
    assert np.array_equal(np.asarray(sponge_del2_profile(30, 0, 1.0)), np.ones(30))
    assert np.array_equal(np.asarray(sponge_del2_profile(30, 3, 1.0)), np.ones(30))
    prof = np.asarray(sponge_del2_profile(30, 3, 8.0))
    np.testing.assert_allclose(prof[:4], [8.0, 4.0, 2.0, 1.0], rtol=1e-12)
    assert np.array_equal(prof[3:], np.ones(27))
    assert MPASPrimitiveEquationConfig().sponge_del2_top_layers == 0
    assert MPASPrimitiveEquationConfig().sponge_del2_top_factor == 1.0


def test_config_defaults_gates_and_wiring():
    assert DycoreConfig().mpas_sponge_del2_top_layers == 0
    assert DycoreConfig().mpas_sponge_del2_top_factor == 1.0
    mpas = dict(discretization="mpas")
    grid = GridConfig(grid_type="mpas", vertical_coord="sigma", nlev=36, resolution=6)
    ExperimentConfig(dycore=DycoreConfig(mpas_sponge_del2_top_layers=3,
                                         mpas_sponge_del2_top_factor=8.0, **mpas),
                     grid=grid).validate_strict()
    for bad in (dict(mpas_sponge_del2_top_layers=-1), dict(mpas_sponge_del2_top_layers=36),
                dict(mpas_sponge_del2_top_factor=0.5)):
        with pytest.raises(ValueError, match="mpas_sponge_del2"):
            ExperimentConfig(dycore=DycoreConfig(**mpas, **bad), grid=grid).validate_strict()
    with pytest.raises(ValueError, match="mpas_sponge_del2"):
        ExperimentConfig(dycore=DycoreConfig(discretization="cdgrid",
                                             mpas_sponge_del2_top_layers=2)).validate_strict()
    args = build_arg_parser().parse_args(["--mpas-sponge-del2-top-layers", "3",
                                          "--mpas-sponge-del2-top-factor", "8"])
    cfg = build_config_from_args(args)
    assert cfg.dycore.mpas_sponge_del2_top_layers == 3
    assert cfg.dycore.mpas_sponge_del2_top_factor == 8.0
