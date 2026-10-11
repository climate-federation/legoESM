from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round246_atomic_pair_gate as gate,
)


def _difference(unequal: int = 0):
    return {
        "support": 1, "unequal": unequal, "signed_zero_only": 0,
        "max_abs": float(unequal), "argmax": [0], "first_nonfinite": None,
    }


def _report():
    direct = {}
    ladders = {}
    operands = {}
    for label in gate.LABELS:
        direct[label] = {
            "status": "PASS_R245_V_DEPTH_OPERAND_SPLIT",
            "raw_v_surface_mask": True,
            "t_pivot_north_neighbor": True,
            "passivity": {"T": True, "S": True},
            "rows": [{
                "substep": step,
                "pair": {
                    name: {"complete": _difference()}
                    for name in (
                        "v_mid", "face_depth_v_mid", "metric_transport_v")
                },
            } for step in (3, 4, 5, 6)],
        }
        rows = []
        for kt in range(1, 11):
            for checkpoint in ("entry", "stage1", "stage2", "stage3"):
                for field in ("T", "S", "u", "v", "ssh"):
                    row = {
                        "kt": kt, "checkpoint": checkpoint, "field": field,
                        "max_abs": 0.0, "rms": 0.0,
                    }
                    if kt == 1 and checkpoint == "stage1" and field in ("T", "S"):
                        row["max_abs"] = 0.001
                    rows.append(row)
        ladders[label] = {
            "last_completed_kt": 10, "checkpoint_count": 40,
            "row_count": 200, "rows": rows,
        }
        operands[label] = {
            "status": "PASS_R232_V_TRANSPORT_OPERAND_SPLIT",
            "passivity": {"T": True, "S": True},
            "replay_checks": {"e3v": _difference(), "zFv": _difference()},
            "operand_rows": {
                name: _difference() for name in (
                    "e1v", "e3v", "vv", "zvb", "vmask")
            },
            "correction_input_rows": {"vn_adv": _difference(35)},
            "zFv_signature": _difference(35),
        }
    return {"direct": direct, "ladders": ladders, "operands": operands}


def test_classifier_accepts_the_frozen_atomic_signature():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R246_ATOMIC_PAIR_THROUGH_KT10"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)


def _held_inputs():
    report = _report()
    direct = report["direct"]
    operands = report["operands"]
    for value in operands.values():
        value["operand_rows"]["vmask"] = _difference(668)
        value["operand_rows"]["vmask"]["max_abs"] = 1.0
        value["first_unequal_operand"] = "vmask"
        value["correction_input_rows"] = {
            name: _difference() for name in ("vn_adv", "r1_hv", "vv_b")
        }
        value["zFv_signature"] = _difference(668)
    logs = {
        label: "\n".join(
            [f"PROGRESS round241 {label} kt={kt}" for kt in range(1, 8)]
            + ["raw-mesh e3w_int must contain only finite values > 0",
               "STATUS REFUSE none: guard fired"])
        for label in gate.LABELS
    }
    return direct, logs, operands


def test_held_classifier_names_stage_mask_and_every_plant_refuses():
    direct, logs, operands = _held_inputs()
    result = gate.classify_held(direct, logs, operands)
    assert result["status"] == "HELD_R246_STAGE_VMASK_AND_KT8_LIVE_W"
    assert result["first_remaining_statement"]["operand"] == (
        "stage-1 three-dimensional V mask")
    for plant in gate.HELD_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify_held(direct, logs, operands, plant=plant)
