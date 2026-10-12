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


def _held_report(label):
    boundaries = []
    for kt in range(1, 5):
        for stage in range(4):
            value = 0.0
            if (kt, stage) == (1, 1):
                value = 0.027227136128767482
            elif (kt, stage) >= (1, 2):
                value = 6.817836451535307
            fields = {}
            for name in gate.r249.FIELDS:
                maximum = (6.817836451535307
                           if (kt, stage) == (1, 2) and name == "S"
                           else value)
                row = {"max_abs": maximum, "nonfinite": 0}
                fields[name] = {
                    "complete": copy.deepcopy(row),
                    "fold_band": copy.deepcopy(row),
                    "interior": copy.deepcopy(row),
                }
            if (kt, stage) == (1, 1):
                fields["T"]["fold_band"]["max_abs"] = 0.0
                fields["S"]["fold_band"]["max_abs"] = 0.0
            boundaries.append({"kt": kt, "stage": stage, "fields": fields})
    return {
        "status": "PASS_R249_STAGE_TABLE", "label": label,
        "unit": True, "operand_route": "resolved-candidate",
        "boundaries": boundaries,
        "passivity": [{"kt": kt, "fields": {"T": True, "S": True}}
                      for kt in range(1, 5)],
    }


def test_held_classifier_and_plants():
    reports = [_held_report(label) for label in gate.LABELS]
    refusal = (
        "PROGRESS round249 independent unit=1 kt=4\n"
        "raw-mesh e3w_int must contain only finite values > 0\n")
    result = gate.classify_held(copy.deepcopy(reports), refusal)
    assert result["status"] == "HELD_R259_KT1_STAGE2_GROWTH"
    for plant in gate.HELD_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify_held(copy.deepcopy(reports), refusal, plant=plant)
