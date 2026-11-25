import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round239_v_rhs_pair_gate as gate,
)


def _row():
    return {
        "at_floor": False, "bit_exact": False, "differing_cells": 68,
        "compared_cells": 68, "absolute_max": 1.0, "rms": 1.0,
        "argmax": [147, 1], "candidate_at_argmax": 1.0,
        "oracle_at_argmax": 0.0,
    }


def _report():
    rows = {name: _row() for name in gate.ORDER}
    first = {"boundary": "after_hpg", **rows["after_hpg"]}
    pair_row = {
        "support": 1050, "unequal": 668, "max_abs": 2.3e-6,
        "argmax": [147, 55, 23], "pair_on_equals_pair_off": True,
        "pair_on_passivity": {"T": True},
        "pair_off_passivity": {"T": True},
    }
    return {
        "pair_completed_rhs": {
            "independent": copy.deepcopy(pair_row),
            "given_nemo_entry": copy.deepcopy(pair_row),
        },
        "operator_table": {
            "source_order": list(gate.ORDER),
            "target": {"cells": 68, "row": 147, "all_on_one_row": True},
            "operator_rows": rows, "first_operator": first,
            "candidate_calibration": {
                "same_over_floor_set": True, "all_faces_at_floor": True,
            },
        },
    }


def test_classifier_holds_at_hpg_after_pair_refutation():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R239_FIRST_V_RHS_OPERATOR_HPG"
    assert result["predictions"]["R239-P2"].startswith("REFUTED")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
