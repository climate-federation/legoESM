"""Direct test for scripts/validate/diag_dycore_only_continuation.py.

The probe attributed the CAM6 arm's warming to the dynamical core, so its
argument surface is load-bearing: a silently-renamed flag would make a future
re-run measure a different thing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_dycore_only_continuation.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_dyncore_only", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_defaults_and_flags():
    a = _load().build_arg_parser().parse_args(["--config", "deck.yaml"])
    assert a.config == "deck.yaml"
    # No --restart means a cold start from the deck's own initial condition.
    assert a.restart is None
    assert a.thermo_terms is False
    assert (a.steps, a.stride, a.hot_level, a.hot_threshold) == (192, 16, 3, 300.0)
    assert a.extra == []


def test_restart_and_passthrough_flags():
    a = _load().build_arg_parser().parse_args(
        ["--config", "deck.yaml", "--restart", "ckpt.npz", "--steps", "64",
         "--stride", "8", "--hot-level", "5", "--hot-threshold", "200",
         "--thermo-terms", "--extra", "--ic-path", "/x.zarr", "--solar-file", "/s.nc"])
    assert a.restart == "ckpt.npz"
    assert a.thermo_terms is True
    assert (a.steps, a.stride, a.hot_level, a.hot_threshold) == (64, 8, 5, 200.0)
    # --extra forwards run_amip's machine-path flags verbatim.
    assert a.extra == ["--ic-path", "/x.zarr", "--solar-file", "/s.nc"]


def test_config_is_required():
    with pytest.raises(SystemExit):
        _load().build_arg_parser().parse_args([])
