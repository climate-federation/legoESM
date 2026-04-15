"""Smoke tests for the analytical ML physics workflow script."""

from __future__ import annotations

from scripts.ml_physics_parameterization import build_arg_parser, _sample_days_from_args


def test_workflow_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.resolution == 8
    assert args.nlev == 40
    assert args.days == 28


def test_sample_day_parsing():
    parser = build_arg_parser()
    args = parser.parse_args(["--sample-days", "0,1,2,7,14,21,28"])
    assert _sample_days_from_args(args) == (0.0, 1.0, 2.0, 7.0, 14.0, 21.0, 28.0)
