"""Fail-closed controls for ORCA2 round 272."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round272_een_leaf_split as gate,
)


def _score(value=0.0, unequal=0):
    return {"max_abs": value, "unequal": unequal}


def _report(label):
    topology = {"count": 100, "coastal": 4, "open": 96, "rows": []}
    return {
        "status": "PASS_R272_MEASURED_REPLAY_REFUSAL",
        "label": label,
        "capture": {"candidate_source_replay": _score(2.0e-9, 10)},
        "compiled_source": {"ln_dynvor_msk": False},
        "topology": topology,
        "relative_circulation": {
            "candidate_vs_nemo_literal": _score(1.0e-8, 2),
            "output_vs_candidate_output": _score(1.0e-8, 2),
            "after_vorticity_vs_nemo": _score(0.0, 0),
        },
        "first_non_floor_leaf": None,
    }


def test_classification_holds_uncalibrated_replay_and_dead_mask():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["first_non_floor_leaf"] is None
    assert result["status"] == "PASS_R272_HELD_REPLAY_UNCALIBRATED"
    assert result["mask_branch_disposition"] == "DEAD_ln_dynvor_msk_FALSE"


def test_classification_plant_refuses():
    reports = [_report(label) for label in gate.LABELS]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=True)


def test_topology_classifier_distinguishes_open_and_coastal():
    residual = np.zeros((148, 90, 30), np.float64)
    residual[10, 20, 3] = 2.0
    residual[30, 40, 3] = 1.0
    u_mask = np.ones((148, 181, 30), bool)
    v_mask = np.ones((149, 180, 30), bool)
    v_mask[30, 40, 3] = False
    rows = gate._topology_rows(residual, u_mask, v_mask, count=2)
    assert rows["open"] == 1
    assert rows["coastal"] == 1
