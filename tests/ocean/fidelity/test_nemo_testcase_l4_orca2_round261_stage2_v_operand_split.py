from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round261_stage2_v_operand_split as gate,
)


def _row(unequal=0, max_abs=0.0):
    return {"unequal": unequal, "max_abs": max_abs, "support": 1,
            "rms": max_abs, "nonfinite": 0}


def _report(label):
    operands = {name: _row() for name in gate.OPERANDS}
    operands["zvb"] = _row(10, 1.0)
    substitutions = {name: _row(10, 1.0) for name in gate.OPERANDS}
    substitutions["zvb"] = _row()
    substitutions["vmask"] = _row()
    return {
        "status": "PASS_R261_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": {"T": True, "S": True},
        "operand_order": list(gate.OPERANDS),
        "correction_input_order": list(gate.CORRECTION_INPUTS),
        "operand_rows": operands,
        "correction_input_rows": {
            name: _row() for name in gate.CORRECTION_INPUTS},
        "first_valued_operand": "zvb",
        "zFv": _row(849, 0.7981690085143782),
        "replay": {"candidate": _row(), "oracle": _row()},
        "cumulative_substitutions": substitutions,
        "round260": _row(849, 0.7981690085143782),
        "one_bit_control": _row(1),
        "records": {"record.bin": "a" * 64},
    }


def test_classifier_names_zvb_and_its_single_variable_closure():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["status"] == "PASS_R261_STAGE2_V_OPERAND_SPLIT"
    assert result["first_valued_operand"] == "zvb"
    assert result["closing_operand"] == "zvb"
    assert result["predictions"]["R261-P3"] == "CONFIRMED"


def test_classifier_retains_a_refuted_first_owner_prediction():
    reports = [_report("independent"), _report("given_nemo_entry")]
    for report in reports:
        report["operand_rows"]["e3v"] = _row(2, 0.5)
        report["first_valued_operand"] = "e3v"
    result = gate.classify(reports)
    assert result["first_valued_operand"] == "e3v"
    assert result["predictions"]["R261-P3"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    reports = [_report("independent"), _report("given_nemo_entry")]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
