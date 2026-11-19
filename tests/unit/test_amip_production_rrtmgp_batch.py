"""The production AMIP deck runs RRTMGP 16 g-points per vmap block (GPU).

Production runs on GPU (user decision 2026-11-18): on one A100 at MPAS res6
the radiation-batched step is 0.132 s against 0.161 s one g-point at a time.
On the CPU nodes the same setting is ~1.7x slower per radiation call
(2026-10-06, res6, 16 ranks x 8 cores: 13.8 s at 16 vs 7.9 s at 0), so a CPU
run passes --rrtmgp-gpoint-batch-size 0.  The deck names the value so it is a
record, not an inherited default.
"""
from __future__ import annotations

import pathlib

_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip" / "amip_production.yaml"


def test_production_deck_resolves_sixteen_gpoints_per_block():
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    keys = load_yaml_config(_CONFIG, parser)
    assert keys["rrtmgp_gpoint_batch_size"] == 16
    parser.set_defaults(**keys)  # same two-pass load as run_amip.main
    assert parser.parse_args([]).rrtmgp_gpoint_batch_size == 16
    # the CPU override the deck comment names must still win over the deck
    assert parser.parse_args(["--rrtmgp-gpoint-batch-size", "0"]).rrtmgp_gpoint_batch_size == 0


def test_code_default_stays_the_gpu_block_size():
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg.rrtmgp_gpoint_batch_size == 16
