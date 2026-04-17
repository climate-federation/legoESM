"""Smoke tests for the analytical ML physics workflow script."""

from __future__ import annotations

from pathlib import Path

from scripts.ml_physics_parameterization import (
    _make_base_config,
    _sample_days_from_args,
    build_arg_parser,
)


def test_workflow_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.resolution == 8
    assert args.nlev == 40
    assert args.days == 28
    assert args.microphysics == "none"


def test_sample_day_parsing():
    parser = build_arg_parser()
    args = parser.parse_args(["--sample-days", "0,1,2,7,14,21,28"])
    assert _sample_days_from_args(args) == (0.0, 1.0, 2.0, 7.0, 14.0, 21.0, 28.0)


def test_kessler_workflow_uses_rrtmgp_base_config():
    parser = build_arg_parser()
    args = parser.parse_args(["--microphysics", "kessler"])
    cfg = _make_base_config(args, Path("results/test_ml_physics"))
    assert cfg.microphysics == "kessler"
    assert cfg.radiation == "rrtmgp"
    assert cfg.cloud_scheme == "xu_randall"
    assert cfg.ozone_source == "analytical"


def test_sundqvist_workflow_uses_rrtmgp_base_config():
    parser = build_arg_parser()
    args = parser.parse_args(["--microphysics", "sundqvist"])
    cfg = _make_base_config(args, Path("results/test_ml_physics"))
    assert cfg.microphysics == "sundqvist"
    assert cfg.radiation == "rrtmgp"
    assert cfg.cloud_scheme == "sundqvist"
    assert cfg.ozone_source == "analytical"
