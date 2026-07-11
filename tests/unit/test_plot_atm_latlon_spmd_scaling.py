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
        {"n_devices": 1, "fused_step_ms": 10.0},
        {"n_devices": 2, "fused_step_ms": 5.0},   # perfect 2x
        {"n_devices": 4, "fused_step_ms": 4.0},   # 2.5x
    ]
    nd, sp = mod.speedup_strong(rows)
    assert nd == [1, 2, 4]
    assert sp[0] == 1.0
    assert abs(sp[1] - 2.0) < 1e-9
    assert abs(sp[2] - 2.5) < 1e-9


def test_efficiency_weak():
    mod = _load()
    rows = [
        {"n_devices": 1, "fused_step_ms": 10.0},
        {"n_devices": 2, "fused_step_ms": 11.0},  # 0.909 efficiency
    ]
    nd, eff = mod.efficiency_weak(rows)
    assert nd == [1, 2]
    assert eff[0] == 1.0
    assert abs(eff[1] - 10.0 / 11.0) < 1e-9


def test_row_time_prefers_contract_key_falls_back_to_legacy():
    """M1 rows carry fused_step_ms; canonical time_per_step_ms is next;
    pre-M1 JSONL with only steady_min_ms must still plot (codex batch4:
    the plotter used to REQUIRE the removed steady_min_ms)."""
    import pytest
    mod = _load()
    assert mod._row_time_ms(
        {"fused_step_ms": 3.0, "time_per_step_ms": 7.0,
         "steady_min_ms": 9.0}) == 3.0
    assert mod._row_time_ms(
        {"time_per_step_ms": 7.0, "steady_min_ms": 9.0}) == 7.0
    assert mod._row_time_ms({"steady_min_ms": 9.0}) == 9.0
    # fused null (zero-length parity row) falls through to the next key.
    assert mod._row_time_ms(
        {"fused_step_ms": None, "time_per_step_ms": 7.0}) == 7.0
    with pytest.raises(KeyError):
        mod._row_time_ms({"n_devices": 2})


def test_empty_inputs_safe():
    mod = _load()
    assert mod.speedup_strong([]) == ([], [])
    assert mod.efficiency_weak([]) == ([], [])
