from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round258_v_transport_pivot_gate as gate,
)


def _row(unequal=0, max_abs=0.0):
    return {"unequal": unequal, "max_abs": max_abs}


def _report(label):
    rows = {name: _row() for name in gate.OPERANDS}
    rows["vmask"] = _row(668, 1.0)
    exact = _row()
    return {
        "status": "PASS_R258_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-geometry-compact-stage-mask",
        "private_unit": True,
        "passivity": {"T": True, "S": True},
        "operand_order": list(gate.OPERANDS),
        "pivot_zfv_support": 668,
        "operand_rows": rows,
        "first_valued_operand": "vmask",
        "candidate_replay": exact.copy(),
        "oracle_replay": exact.copy(),
        "baseline_stage": {
            "T": _row(668, 0.17735225873138316),
            "S": _row(668, 3.283688696376231),
        },
        "mask_arm": {"vmask": exact.copy(), "zFv": exact.copy()},
        "mask_arm_stage": {"T": exact.copy(), "S": exact.copy()},
        "round247_reconciliation": {
            "record_sha256": "same", "round247_record_sha256": "same",
            "round247_vmask_unequal": 0, "round247_zfv_unequal": 0,
            "current_vmask_unequal": 0, "current_zfv_unequal": 0,
        },
        "one_bit_control": {"unequal": 1},
    }


def test_classifier_names_compact_stage_vmask_and_preserves_both_labels():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["first_valued_operand"] == "vmask"
    assert result["predictions"]["R258-P4"] == "CONFIRMED_PRIVATE_ARM"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    reports = [_report("independent"), _report("given_nemo_entry")]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
