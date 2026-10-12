from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round259_atomic_unit_gate as gate,
)


def _row(value=0.0):
    return {"unequal": 0, "max_abs": value, "rms": value,
            "nonfinite": 0}


def _report(label):
    boundaries = []
    for kt in range(1, 11):
        for stage in range(4):
            value = 0.0 if (kt, stage) in ((1, 0), (1, 1)) else 1.0e-9
            boundaries.append({
                "kt": kt, "stage": stage,
                "fields": {
                    name: {"fold_band": _row(value),
                           "interior": _row(value)}
                    for name in gate.r249.FIELDS},
            })
    return {
        "status": "PASS_R259_MEASUREMENT", "label": label,
        "pivot_transport": {
            "status": "PASS_R259_PIVOT_TRANSPORT",
            "execution": "production-jit-cpu-fp64-x64-libm",
            "operand_route": "production-d114-resolved-stage-vmask",
            "private_unit": True, "passivity": {"T": True, "S": True},
            "pivot_support": 668, "vmask": _row(), "zFv": _row(),
        },
        "stage_growth": {
            "status": "PASS_R249_STAGE_TABLE", "unit": True,
            "operand_route": "resolved-candidate",
            "passivity": [{"fields": {"T": True}} for _ in range(10)],
            "boundaries": boundaries,
        },
        "one_bit_control": {"unequal": 1},
    }


def test_classifier_accepts_both_labels_through_kt10():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["status"] == "PASS_R259_COMPLETE_UNIT_KT10"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    reports = [_report("independent"), _report("given_nemo_entry")]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
