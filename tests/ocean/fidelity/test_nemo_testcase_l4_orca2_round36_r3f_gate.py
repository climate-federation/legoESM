"""Binding tests for the ORCA2 round-36 native-F-area gate."""

from types import SimpleNamespace

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round36_r3f_area_gate as gate,
)


def _replay(status, u_max, v_max, exact):
    return {
        "status": status,
        "u_momentum": {"max_abs": u_max, "bit_identical": exact},
        "v_momentum": {"max_abs": v_max, "bit_identical": exact},
        "worktree": {"commit": "test"},
    }


def _boundary(*, unshifted_exact=True):
    return {
        "area_f_grid_vs_raw_product": {
            "unequal": gate.EXPECTED_SHIFTED_UNEQUAL,
            "max_abs": gate.EXPECTED_SHIFTED_MAX,
        },
        "area_f_unshifted_grid_vs_raw_product": {
            "bit_identical": unshifted_exact,
        },
    }


def test_gate_requires_exact_mapping_and_replay(monkeypatch):
    responses = iter((
        _replay("HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
                gate.EXPECTED_PARENT_MAX["v_momentum"], False),
        _replay("PASS", 0.0, 0.0, True),
    ))
    monkeypatch.setattr(gate.r35, "capture_r3f_boundary",
                        lambda *args, **kwargs: _boundary())
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "AT_BAR"


def test_gate_rejects_shifted_native_mapping(monkeypatch):
    responses = iter((
        _replay("HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
                gate.EXPECTED_PARENT_MAX["v_momentum"], False),
        _replay("PASS", 0.0, 0.0, True),
    ))
    monkeypatch.setattr(gate.r35, "capture_r3f_boundary",
                        lambda *args, **kwargs: _boundary(unshifted_exact=False))
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "DEBT"


def test_gate_rejects_one_remaining_replay_bit(monkeypatch):
    responses = iter((
        _replay("HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
                gate.EXPECTED_PARENT_MAX["v_momentum"], False),
        _replay("HELD", float.fromhex("0x1p-1074"), 0.0, False),
    ))
    monkeypatch.setattr(gate.r35, "capture_r3f_boundary",
                        lambda *args, **kwargs: _boundary())
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "DEBT"
