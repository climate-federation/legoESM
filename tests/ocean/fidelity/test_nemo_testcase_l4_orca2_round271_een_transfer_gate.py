import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round271_een_transfer_gate as gate,
)


def _score(value):
    return {"max_abs": value, "rms": value / 10.0, "unequal": 3,
            "argmax": [19, 33, 3]}


def _report(label, value):
    at_floor = {name: _score(0.0) for name in ("u", "v", "T", "S", "eta")}
    return {
        "label": label,
        "status": "PASS_R266_MEASUREMENT",
        "boundary_rows": {"after_vorticity_u": {"complete": _score(value)}},
        "accumulator_self_replay": {"after_vorticity_u": _score(0.0)},
        "matched_stage1": at_floor,
    }


def test_gate_keeps_the_failed_prediction_and_plant_fires():
    before = [_report(label, 1.3e-8) for label in gate.LABELS]
    candidate = [_report(label, 1.2e-8) for label in gate.LABELS]
    result = gate.classify(before, candidate)
    assert result["status"] == "PASS_R271_HELD_WRITTEN_ORDER_NOT_CLOSED"
    assert result["predictions"]["R271-P2"].startswith("REFUTED")
    assert result["predictions"]["R271-P6"] == (
        "CONFIRMED_BRANCH_LANDED_AFTER_MEASUREMENT")
    with pytest.raises(gate.GateError, match="unexpectedly closed"):
        gate.classify(before, candidate, plant=True)


def test_gate_refuses_a_moved_label_or_nonpassive_accumulator():
    before = [_report(label, 1.3e-8) for label in gate.LABELS]
    candidate = [_report(label, 1.2e-8) for label in gate.LABELS]
    bad = copy.deepcopy(candidate)
    bad[1]["boundary_rows"]["after_vorticity_u"]["complete"]["max_abs"] = 1.1e-8
    with pytest.raises(gate.GateError, match="labels disagree"):
        gate.classify(before, bad)
    bad = copy.deepcopy(candidate)
    bad[0]["accumulator_self_replay"]["after_vorticity_u"]["max_abs"] = 1.0
    with pytest.raises(gate.GateError, match="self-replays"):
        gate.classify(before, bad)
