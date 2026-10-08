from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round179_v_rhs_operator_walk as gate,
)


def _row(*, bit_exact: bool, at_floor: bool) -> dict[str, object]:
    return {
        "bit_exact": bit_exact, "at_floor": at_floor,
        "differing_cells": 0 if bit_exact else 1,
    }


def _report() -> dict[str, object]:
    rows = {name: _row(bit_exact=True, at_floor=True) for name in gate.BOUNDARIES}
    rows["after_hpg"] = _row(bit_exact=False, at_floor=False)
    first = {"boundary": "after_hpg", **rows["after_hpg"]}
    return {
        "claim_label": "independent hierarchy rung 0",
        "record_census": {
            name: {"coverage": "exactly-once"}
            for name in ("rhs", "slow", "static", "spg")
        },
        "record_control": {"bit_exact": False, "differing_cells": 1},
        "source_order": list(gate.BOUNDARIES),
        "operand_order": list(gate.OPERANDS),
        "arm_order": list(gate.ARMS),
        "cross_record_closure": {
            "after_zad_to_depth_v": {"bit_exact": True},
            "depth_v_to_completed_v": {"bit_exact": True},
        },
        "candidate_calibration": {
            "same_over_floor_set": True, "all_faces_at_floor": True},
        "target": {"cells": 68, "row": 147, "all_on_one_row": True},
        "operator_rows": rows, "first_operator": first,
        "operand_rows": {
            "raw_hpg_rhs": {"bit_exact": True},
            "e3v": {"bit_exact": True},
            "vmask": {"bit_exact": False},
            "r1_hv0": {"bit_exact": True},
        },
        "first_operand": "vmask",
        "operand_arms": {
            "oracle": {"bit_exact": True},
            "candidate_e3v": {"bit_exact": True},
            "candidate_vmask": {"bit_exact": False},
            "candidate_r1_hv0": {"bit_exact": True},
        },
        "mask_only_reproduces_candidate_hpg": True,
        "endpoint_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_classifier_accepts_registered_hpg_mask_result():
    result = gate.classify(_report())
    assert result["status"] == "HELD_FIRST_V_RHS_OPERAND"
    assert result["prediction_ledger"] == {
        "R179-P1": "CONFIRMED", "R179-P2": "CONFIRMED",
        "R179-P3": "CONFIRMED", "R179-P4": "CONFIRMED",
        "R179-P5": "CONFIRMED",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_first_operator_follows_compiled_order():
    report = _report()
    report["operator_rows"]["after_hpg"] = _row(bit_exact=True, at_floor=True)
    report["operator_rows"]["after_ldf"] = _row(bit_exact=False, at_floor=False)
    report["first_operator"] = {
        "boundary": "after_ldf", **report["operator_rows"]["after_ldf"]}
    result = gate.classify(report)
    assert result["prediction_ledger"]["R179-P3"] == "REFUTED"
