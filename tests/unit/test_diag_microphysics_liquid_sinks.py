"""Direct test for scripts/validate/diag_microphysics_liquid_sinks.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_microphysics_liquid_sinks.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_sinks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flags():
    m = _load()
    a = m.build_arg_parser().parse_args(["--config", "d.yaml", "--restart", "c.npz"])
    assert a.band == [500.0, 800.0] and a.dt_split is False
    with pytest.raises(SystemExit):
        m.build_arg_parser().parse_args(["--restart", "c.npz"])


def test_weighted_rate_is_area_and_mass_weighted():
    m = _load()
    rate = np.array([[1.0, 2.0], [4.0, 0.0]])
    dp = np.full((2, 2), 100.0)
    w = np.array([0.25, 0.75])
    assert abs(m.weighted_rate(rate, dp, w, 10.0) - 37.5) < 1e-12


def test_masked_rate_selects_only_the_masked_layers():
    m = _load()
    rate = np.array([[1.0, 2.0]])
    dp = np.full((1, 2), 100.0)
    w = np.array([1.0])
    mask = np.array([[True, False]])
    assert abs(m.masked_rate(rate, dp, w, 10.0, mask) - 10.0) < 1e-12
    allm = np.array([[True, True]])
    assert abs(m.masked_rate(rate, dp, w, 10.0, allm)
               - m.weighted_rate(rate, dp, w, 10.0)) < 1e-12
