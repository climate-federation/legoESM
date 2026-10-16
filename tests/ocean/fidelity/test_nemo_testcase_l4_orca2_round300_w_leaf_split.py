"""Known-answer controls for round 300's W-leaf classifier."""

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round300_w_leaf_split as gate,
)


def _score(value):
    row = {"max_abs": value, "unequal": int(value != 0.0)}
    return {"complete": dict(row), "interior": dict(row), "fold_band": dict(row)}


def _label(name):
    rows = {leaf: _score(0.0) for leaf in gate.SOURCE_ORDER}
    rows["u_difference"] = _score(1.0)
    rows["v_difference"] = _score(2.0)
    return {
        "status": "PASS_R300_LABEL_MEASUREMENT", "label": name,
        "passivity": {"T": True, "S": True}, "rows": rows,
        "candidate_replay": {"ww": _score(0.0), "trp_w": _score(0.0)},
        "oracle_replay": {"ww": _score(0.0), "trp_w": _score(0.0)},
        "cumulative_arms": {
            "trp_u": _score(1.0),
            "trp_u+trp_v": _score(0.0),
        },
        "one_bit_control": {"unequal": 1},
    }


def _report():
    return {
        "admission": {"status": "PASS_R299_OMT4_KT7_W_ADMISSION"},
        "source_order": list(gate.SOURCE_ORDER),
        "labels": [_label(name) for name in gate.LABELS],
    }


def test_classifier_names_u_difference_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["classification"]["first_nonbit_statement"] == "u_difference"
    assert result["classification"]["prediction_ledger"]["R300-P3"] == "CONFIRMED"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(copy.deepcopy(_report()), plant)


def test_program_reuses_source_helpers_and_zeroes_the_bottom():
    shape = (2, 3, 2)
    inputs = {
        "trp_u": np.zeros(shape), "trp_v": np.zeros(shape),
        "w_r1area": np.ones(shape[:2]), "w_area": np.ones(shape[:2]),
        "w_e3t0": np.ones(shape), "w_r3bb": np.zeros(shape[:2]),
        "w_r3mm": np.zeros(shape[:2]), "w_r3aa": np.zeros(shape[:2]),
        "w_tmask": np.ones(shape),
    }
    result = gate._program(inputs, 1.0)
    assert tuple(result) == gate.SOURCE_ORDER
    assert np.array_equal(result["w_recurrence"], np.zeros((2, 3, 3)))
    assert np.array_equal(result["area_product"], np.zeros((2, 3, 3)))


def test_cumulative_walk_is_monotone_in_recorded_operand_sets():
    previous = set()
    for arm in gate.CUMULATIVE_ARMS:
        current = set(arm)
        assert previous < current
        previous = current
