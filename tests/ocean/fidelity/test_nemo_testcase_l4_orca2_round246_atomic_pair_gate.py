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
            "status": "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
            "passivity": {"T": True, "S": True},
            "operand_rows": {
                name: _difference() for name in (
                    "e3v", "vmask", "vn_adv", "r1_hv", "zvb", "zFv")
            },
            "fold_band": {"T": _difference(), "S": _difference()},
        }
        operands[label]["operand_rows"]["vn_adv"] = _difference(35)
        operands[label]["operand_rows"]["zFv"] = _difference(35)
    return {"direct": direct, "ladders": ladders, "operands": operands}


def test_classifier_accepts_the_frozen_atomic_signature():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R246_ATOMIC_PAIR_THROUGH_KT10"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
