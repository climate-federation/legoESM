"""Direct test for scripts/validate/diag_clubb_liquid_handoff.py.

The probe measured how much of the CLUBB closure's liquid never reaches the
cloud optics, and how much of the baseline's reflectivity the diagnostic
condensate floor supplies.  Its argument surface is load-bearing: a renamed
flag would make a re-run measure a different state or a different deck.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_clubb_liquid_handoff.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_clubb_handoff", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_required_flags_and_defaults():
    a = _load().build_arg_parser().parse_args(
        ["--config", "deck.yaml", "--restart", "ckpt.npz"])
    assert (a.config, a.restart) == ("deck.yaml", "ckpt.npz")
    assert a.label == "arm"
    assert a.extra == []


def test_restart_is_required():
    """Without a checkpoint there is no state to measure, so it cannot default."""
    with pytest.raises(SystemExit):
        _load().build_arg_parser().parse_args(["--config", "deck.yaml"])


def test_passthrough_flags_survive():
    a = _load().build_arg_parser().parse_args(
        ["--config", "deck.yaml", "--restart", "ckpt.npz", "--label", "CAM6",
         "--extra", "--solar-file", "/s.nc", "--resolution", "6"])
    assert a.label == "CAM6"
    assert a.extra == ["--solar-file", "/s.nc", "--resolution", "6"]
