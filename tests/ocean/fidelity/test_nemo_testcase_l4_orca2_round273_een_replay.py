"""Classification controls for the round-273 fused EEN replay."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round273_een_relative_replay as gate,
)


def _score(value=0.0):
    return {"max_abs": value, "unequal": 0}


def _report(label):
    return {
        "status": "PASS_R273_FUSED_REPLAY_CALIBRATED",
        "label": label,
        "passivity": {"u": True, "v": True},
        "baseline_after_vorticity_u": _score(1.2863307536456779e-8),
        "candidate_replay": _score(0.0),
        "accumulator_replay": _score(0.0),
    }


def test_classification_requires_both_calibrated_labels():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "PASS_R273_FUSED_REPLAY_CALIBRATED"
    assert result["leaf_verdict"] == "UNMEASURED_REQUIRES_NEMO_LEAF_STREAM"


def test_classification_plant_refuses():
    reports = [_report(label) for label in gate.LABELS]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=True)
