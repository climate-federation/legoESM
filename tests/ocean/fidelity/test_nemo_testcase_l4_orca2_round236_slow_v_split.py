import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round236_slow_v_split as gate,
)


def _report(label):
    exact = {"bit_exact": True, "at_floor": True, "unequal": 0,
             "scored": 35, "max_abs": 0.0, "argmax": [0, 0]}
    rows = {name: copy.deepcopy(exact) for name in gate.SOURCE_ORDER}
    rows["coriolis_v"] = {**exact, "bit_exact": False, "unequal": 35,
                           "max_abs": 1.0e-6, "at_floor": False}
    return {
        "status": "PASS_R236_SLOW_V_SPLIT",
        "label": label,
        "passivity": {"T": True, "S": True, "u": True,
                       "v": True, "eta": True},
        "source_order": list(gate.SOURCE_ORDER),
        "owner_support": {"unequal": 35},
        "rows": rows,
        "first_unequal": "coriolis_v",
        "depth_replay": {"through_reciprocal": exact},
        "final_replay": {"through_incoming": {**exact, "bit_exact": False},
                         "through_mask": exact},
    }


def test_classification_accepts_both_labels():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["status"] == "HELD_R236_SLOW_V_OWNER_NAMED"
    assert result["predictions"]["R236-P2"] == "CONFIRMED_CORIOLIS_REMOVAL_UNIT"


@pytest.mark.parametrize("plant", gate.CLASSIFY_PLANTS[1:])
def test_classification_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify([
            _report("independent"), _report("given_nemo_entry")], plant=plant)


def test_labels_must_name_same_operand():
    given = _report("given_nemo_entry")
    given["first_unequal"] = "completed_v_rhs"
    with pytest.raises(gate.GateError, match="labels disagree"):
        gate.classify([_report("independent"), given])
