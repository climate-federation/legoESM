"""Unit test for the lat-band SPMD scaling plotter's pure metric math
(speedup_strong / efficiency_weak). Rendering is exercised on-cluster."""
from __future__ import annotations

import importlib.util
import os

_PLOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "plot",
    "plot_atm_latlon_spmd_scaling.py")


def _load():
    spec = importlib.util.spec_from_file_location("_plot_spmd", _PLOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_speedup_strong():
    mod = _load()
    rows = [
        {"n_devices": 1, "steady_min_ms": 10.0},
        {"n_devices": 2, "steady_min_ms": 5.0},   # perfect 2x
        {"n_devices": 4, "steady_min_ms": 4.0},   # 2.5x
    ]
    nd, sp = mod.speedup_strong(rows)
    assert nd == [1, 2, 4]
    assert sp[0] == 1.0
    assert abs(sp[1] - 2.0) < 1e-9
    assert abs(sp[2] - 2.5) < 1e-9


def test_efficiency_weak():
    mod = _load()
    rows = [
        {"n_devices": 1, "steady_min_ms": 10.0},
        {"n_devices": 2, "steady_min_ms": 11.0},  # 0.909 efficiency
    ]
    nd, eff = mod.efficiency_weak(rows)
    assert nd == [1, 2]
    assert eff[0] == 1.0
    assert abs(eff[1] - 10.0 / 11.0) < 1e-9


def test_empty_inputs_safe():
    mod = _load()
    assert mod.speedup_strong([]) == ([], [])
    assert mod.efficiency_weak([]) == ([], [])
