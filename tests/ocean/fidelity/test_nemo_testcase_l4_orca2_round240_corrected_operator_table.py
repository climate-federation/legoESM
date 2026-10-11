import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round240_corrected_operator_table as gate,
)


def _row(*, exact):
    return {
        "bit_exact": exact, "at_floor": exact,
        "differing_cells": 0 if exact else 68, "compared_cells": 68,
        "absolute_max": 0.0 if exact else 1.0, "rms": 0.0 if exact else 1.0,
        "argmax": [147, 0], "candidate_at_argmax": 0.0,
        "oracle_at_argmax": 0.0 if exact else 1.0,
    }


def _report():
    rows = {name: _row(exact=True) for name in gate.ORDER}
    rows["after_ldf"] = _row(exact=False)
    rows["after_vor"] = _row(exact=False)
    rows["after_keg"] = _row(exact=False)
    rows["after_zad"] = _row(exact=False)
    return {
        "claim_label": "independent hierarchy rung 0",
        "source_order": list(gate.ORDER),
        "geometry_route": "nemo_qco_resolved_mesh_operands",
        "target": {"cells": 68, "row": 147},
        "trace_passivity": {"T": True, "S": True},
        "candidate_calibration": {
            "all_faces_at_floor": True, "same_over_floor_set": True,
        },
        "operator_rows": rows,
        "first_operator": {"boundary": "after_ldf", **rows["after_ldf"]},
    }


def test_classifier_names_the_first_downstream_operator():
    assert gate.classify(_report())["status"] == "HELD_R240_FIRST_OPERATOR_AFTER_LDF"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
