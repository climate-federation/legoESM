"""Binding tests for the ORCA2 round-37 measured-pair gate."""

from types import SimpleNamespace

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round37_r3f_pair_gate as gate,
)


def _replay(status, u_max, v_max, u_unequal, v_unequal, exact):
    return {
        "status": status,
        "u_momentum": {"max_abs": u_max, "unequal": u_unequal,
                       "bit_identical": exact},
        "v_momentum": {"max_abs": v_max, "unequal": v_unequal,
                       "bit_identical": exact},
        "worktree": {"commit": "test"},
    }


def _responses(*, pair_exact=True):
    parent = _replay(
        "HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
        gate.EXPECTED_PARENT_MAX["v_momentum"], 410460, 407570, False)
    reciprocal = _replay(
        "HELD", gate.EXPECTED_PARENT_MAX["u_momentum"],
        gate.EXPECTED_PARENT_MAX["v_momentum"], 410460, 407570, False)
    area = _replay(
        "HELD", gate.EXPECTED_ISOLATED_MAX, gate.EXPECTED_ISOLATED_MAX,
        gate.EXPECTED_ISOLATED_UNEQUAL, gate.EXPECTED_ISOLATED_UNEQUAL, False)
    pair = _replay("PASS" if pair_exact else "HELD",
                   0.0 if pair_exact else float.fromhex("0x1p-1074"),
                   0.0, 0 if pair_exact else 1, 0, pair_exact)
    return iter((parent, reciprocal, area, pair))


def test_gate_requires_both_controls_and_exact_pair(monkeypatch):
    responses = _responses()
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "AT_BAR"


def test_gate_rejects_one_remaining_pair_bit(monkeypatch):
    responses = _responses(pair_exact=False)
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "DEBT"


def test_gate_rejects_a_changed_isolated_control(monkeypatch):
    responses = list(_responses())
    responses[2]["u_momentum"]["unequal"] = 23
    responses = iter(responses)
    monkeypatch.setattr(gate.r32, "capture_ldf_replay",
                        lambda *args, **kwargs: next(responses))
    assert gate.capture(SimpleNamespace(), SimpleNamespace())["status"] == "DEBT"
