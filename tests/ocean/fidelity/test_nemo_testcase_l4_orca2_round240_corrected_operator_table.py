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
    rows = {name: _row(exact=False) for name in gate.ORDER}
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
        "first_operator": {"boundary": "after_hpg", **rows["after_hpg"]},
        "raw_hpg": _row(exact=True),
        "compact_mask": _row(exact=False),
        "masked_hpg_to_stage_component": _row(exact=True),
        "first_statement": {
            "boundary": "extra_compact_vmask_after_hpg",
            "candidate": "dv_dt = dv_dt * v_mask_3d",
            "oracle": "pvv=zhpj+zvap then stp2d applies stored vmask in the depth sum",
        },
    }


def test_classifier_names_the_extra_compact_mask():
    assert gate.classify(_report())["status"] == (
        "HELD_R240_FIRST_STATEMENT_EXTRA_COMPACT_VMASK")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
