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
from run_amip import build_arg_parser, build_config_from_args  # noqa: E402


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


def test_run_amip_sigma_top_reaches_the_config():
    args = build_arg_parser().parse_args(["--sigma-top", "0.002"])
    assert args.sigma_top == 0.002
    assert build_arg_parser().parse_args([]).sigma_top is None
    assert build_config_from_args(args).grid.sigma_top == 0.002


def test_yaml_sigma_top_and_refine_reach_the_config(tmp_path):
    from legoesm.config import Config
    yml = tmp_path / "deck.yaml"
    yml.write_text("grid:\n  type: mpas\n  vertical_coord: sigma\n  n_levels: 30\n"
                   "  sigma_top: 0.002\n  tropopause_refine: 3.0\n")
    g = Config.from_yaml(str(yml)).to_experiment_config().grid
    assert g.sigma_top == 0.002 and g.tropopause_refine == 3.0


def test_driver_passes_the_lid_to_the_coordinate():
    import inspect
    from legoesm.driver.model_driver import ModelDriver
    src = inspect.getsource(ModelDriver._create_grid)
    assert "sigma_top=gc.sigma_top" in src


def test_sigma_refine_centre_and_width_thread_and_put_layers_in_the_boundary_layer():
    """--sigma-refine / --sigma-refine-width (2026-09-16) reach GridConfig and the
    coordinate the driver builds; a bump centred at sigma 0.95 with width 0.06
    and refine 3 puts >= 5 of 30 layers above sigma 0.9 (uniform: 3), while
    the defaults reproduce the uniform grid bit for bit at refine 1."""
    parser = build_arg_parser()
    cfg = build_config_from_args(parser.parse_args([
        "--dataset", "analytical", "--vertical-coord", "sigma", "--nlev", "30",
        "--tropopause-refine", "3", "--sigma-refine", "0.95", "--sigma-refine-width", "0.06"]))
    assert cfg.grid.sigma_refine == 0.95 and cfg.grid.sigma_refine_width == 0.06
    cfg.validate_strict()
    with pytest.raises(ValueError, match="sigma_refine"):
        cfg._replace(grid=cfg.grid._replace(sigma_refine=1.5)).validate_strict()
    bl = create_sigma_coordinate(30, tropopause_refine=3.0, sigma_refine=0.95, refine_width=0.06)
    uni = create_sigma_coordinate(30)
    n_bl = int(jnp.sum(jnp.asarray(bl.sigma_full) > 0.9))
    n_uni = int(jnp.sum(jnp.asarray(uni.sigma_full) > 0.9))
    assert n_uni == 3 and n_bl >= 5, (n_uni, n_bl)
    same = create_sigma_coordinate(30, tropopause_refine=1.0, sigma_refine=0.95, refine_width=0.06)
    assert jnp.array_equal(same.sigma_half, uni.sigma_half)
