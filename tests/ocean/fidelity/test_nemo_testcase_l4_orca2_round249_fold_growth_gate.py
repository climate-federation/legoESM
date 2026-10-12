from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round249_fold_growth_gate as gate,
)


def _metric(maximum=0.0):
    return {
        "support": 1, "unequal": int(maximum != 0.0), "max_abs": maximum,
        "rms": maximum, "argmax": [0], "first_nonfinite": None,
        "nonfinite": 0,
    }


def _fold(name, unequal=0):
    return {
        "point_type": gate.POINT_TYPES[name], "sign": gate.SIGNS[name],
        "target_row": 147, "target_i_start": 0, "source_row": 146,
        **_metric(float(unequal)),
    }


def _stage_report(label, unit):
    boundaries = []
    for kt in range(1, 5):
        for stage in range(4):
            fields = {}
            for name in gate.FIELDS:
                maximum = 1.0 if unit and kt == 1 and stage == 1 else 0.0
                fields[name] = {
                    "complete": _metric(maximum),
                    "interior": _metric(0.0),
                    "fold_band": _metric(maximum),
                    "fold_identity": _fold(
                        name, int(unit and kt == 1 and stage == 1)),
                    "nemo_fold_identity": _fold(name),
                }
            boundaries.append({"kt": kt, "stage": stage, "fields": fields})
    return {
        "status": "PASS_R249_STAGE_TABLE", "label": label, "unit": unit,
        "operand_route": "resolved-candidate" if unit else "production",
        "passivity": [
            {"kt": kt, "fields": {name: True for name in gate.FIELDS}}
            for kt in range(1, 5)
        ],
        "boundaries": boundaries,
    }


def _substep_report(label):
    first = {
        "substep": 1, "source_order": 0, "name": "slow_v", "face": "v",
        "complete": _metric(1.0), "interior": _metric(0.0),
        "fold_band": _metric(1.0),
    }
    return {
        "status": "PASS_R249_SUBSTEP_TABLE", "label": label,
        "operand_route": "resolved-candidate",
        "passivity": {name: True for name in gate.FIELDS},
        "source_order": list(gate.r235.SOURCE_ORDER),
        "first_complete": copy.deepcopy(first), "first_interior": None,
        "first_fold_band": copy.deepcopy(first),
    }


def _reports():
    off = {label: _stage_report(label, False) for label in gate.LABELS}
    unit = {label: _stage_report(label, True) for label in gate.LABELS}
    substeps = {label: _substep_report(label) for label in gate.LABELS}
    return off, unit, substeps


def test_classifier_names_the_first_stage_fold_owner_candidate():
    result = gate.classify(*_reports())
    assert result["status"] == "HELD_R249_FOLD_ASSOCIATION_OWNER_CANDIDATE"
    assert all((row["kt"], row["stage"]) == (1, 1)
               for row in result["first_growth"].values())


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round249_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(*_reports(), plant=plant)
