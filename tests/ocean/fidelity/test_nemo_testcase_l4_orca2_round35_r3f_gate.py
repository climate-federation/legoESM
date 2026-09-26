"""Binding tests for the ORCA2 round-35 reciprocal-order gate."""

from types import SimpleNamespace

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round35_r3f_reciprocal_gate as gate,
)


def _replay(status, u_max, v_max, exact):
    return {
        "status": status,
        "u_momentum": {"max_abs": u_max, "bit_identical": exact},
        "v_momentum": {"max_abs": v_max, "bit_identical": exact},
        "worktree": {"commit": "test"},
    }


def test_gate_requires_parent_reproduction_and_an_exact_arm(monkeypatch):
    responses = iter((
        _replay("HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
                gate.EXPECTED_PARENT_MAX["v_momentum"], False),
        _replay("PASS", 0.0, 0.0, True),
    ))
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "AT_BAR"


def test_gate_rejects_one_remaining_arm_bit(monkeypatch):
    responses = iter((
        _replay("HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
                gate.EXPECTED_PARENT_MAX["v_momentum"], False),
        _replay("HELD", float.fromhex("0x1p-1074"), 0.0, False),
    ))
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "DEBT"
