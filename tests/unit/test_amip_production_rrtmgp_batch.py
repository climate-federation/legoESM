"""The production AMIP deck runs RRTMGP one g-point at a time on CPU.

The code default (16 g-points per vmap block) is the GPU throughput setting;
on the CPU nodes it is ~1.7x slower per radiation call (measured 2026-10-06,
res6, 16 ranks x 8 cores: 13.8 s at 16 vs 7.9 s at 0).  The deck names the
CPU value so it is a record, not an inherited default.
"""
from __future__ import annotations

import pathlib

_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip" / "amip_production.yaml"


def test_production_deck_resolves_one_gpoint_at_a_time():
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    keys = load_yaml_config(_CONFIG, parser)
    assert keys["rrtmgp_gpoint_batch_size"] == 0
    parser.set_defaults(**keys)  # same two-pass load as run_amip.main
    assert parser.parse_args([]).rrtmgp_gpoint_batch_size == 0


def test_code_default_stays_the_gpu_block_size():
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg.rrtmgp_gpoint_batch_size == 16


def test_production_deck_names_all_gpoint_gas_optics():
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser
    assert load_yaml_config(_CONFIG, build_arg_parser())["rrtmgp_gas_optics_mode"] == "all_gpts"


def test_gas_optics_mode_flag_parses_and_refuses_unknown():
    import pytest

    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    assert parser.parse_args(["--rrtmgp-gas-optics-mode", "per_gpt"]).rrtmgp_gas_optics_mode == "per_gpt"
    with pytest.raises(SystemExit):
        parser.parse_args(["--rrtmgp-gas-optics-mode", "nonsense"])


def test_experiment_gas_optics_mode_reaches_the_solver_config():
    from legoesm.driver.config import ExperimentConfig
    fields = ExperimentConfig().rrtmgp_gas_optics_mode
    assert fields == "all_gpts"
