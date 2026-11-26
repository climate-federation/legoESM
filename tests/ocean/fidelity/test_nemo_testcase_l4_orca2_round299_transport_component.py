"""Known-answer controls for round 299's transport-component classifier."""

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round299_transport_component as gate,
)


def _regions(value):
    row = {"max_abs": value, "unequal": int(value != 0.0)}
    return {"complete": dict(row), "interior": dict(row), "fold_band": dict(row)}


def _label(name):
    return {
        "status": "PASS_R299_LABEL_MEASUREMENT", "label": name,
        "passivity": {"T": True, "S": True},
        "entry": {"T": _regions(1.0), "S": _regions(2.0)},
        "candidate_replay": {"T": _regions(0.0), "S": _regions(0.0)},
        "direct_replay": {"T": _regions(0.0), "S": _regions(0.0)},
        "candidate": {"T": _regions(0.0), "S": _regions(40.0)},
        "full_nemo": {"T": _regions(0.0), "S": _regions(2.0)},
        "component_arms": {
            "trp_u": {"T": _regions(0.0), "S": _regions(30.0)},
            "trp_v": {"T": _regions(0.0), "S": _regions(2.0)},
            "trp_w": {"T": _regions(0.0), "S": _regions(3.0)},
        },
        "one_bit_control": {"unequal": 1},
    }


def _report():
    return {
        "admission": {"status": "PASS_R297_OMT4_KT7_STAGE1_ADMISSION"},
        "component_order": list(gate.COMPONENTS),
        "labels": [_label(name) for name in gate.LABELS],
    }


def test_classifier_names_v_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["classification"]["first_causal_component"] == "trp_v"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(copy.deepcopy(_report()), plant)


def test_component_geometry_changes_only_the_named_native_transport():
    candidate = tuple(np.full((1,), float(index)) for index in range(12))
    oracle = tuple(np.full((1,), 100.0 + index) for index in range(12))
    expected = {"trp_u": {0, 7}, "trp_v": {1, 8}, "trp_w": {2}}
    for name, changed in expected.items():
        result = gate._component_geometry(candidate, oracle, name)
        assert {index for index, (before, after) in enumerate(zip(candidate, result))
                if not np.array_equal(before, after)} == changed


def test_no_single_component_is_reported_explicitly():
    report = _report()
    for row in report["labels"]:
        for arm in row["component_arms"].values():
            arm["S"]["fold_band"]["max_abs"] = 30.0
    result = gate.classify(report)
    assert result["classification"]["first_causal_component"] == "none"
