"""Known-answer controls for the ORCA2 round-198 offline split."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round198_pressure_trend_split as gate,
)


def _row(bit_exact: bool, cells: int = 0) -> dict[str, object]:
    return {
        "bit_exact": bit_exact,
        "differing_cells": cells,
        "first_unequal_index": None if bit_exact else [0, 0],
        "maximum_absolute": 0.0 if bit_exact else 1.0,
    }


def _report() -> dict[str, object]:
    exact = _row(True)
    debt = _row(False, 3)
    back_inputs = {name: copy.deepcopy(exact) for name in gate.BACK_ORDER}
    back_inputs["ssha_e"] = copy.deepcopy(debt)
    coefficients = {
        name: copy.deepcopy(exact)
        for name in ("ffv_sw", "ffv_se", "ffv_nw", "ffv_ne")
    }
    drag_inputs = {
        name: copy.deepcopy(exact) for name in ("zCdU_v", "vn_e", "hvr_e")
    }
    drag_inputs["zCdU_v"] = copy.deepcopy(debt)
    return {
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": {name: True for name in gate.RECORD_FIELDS},
        "passivity": {"ssh": True, "u": True, "v": True},
        "split": {
            "back_order": list(gate.BACK_ORDER),
            "trend_order": list(gate.TREND_ORDER),
            "pressure": {
                "back_inputs": back_inputs,
                "candidate_back_replay_vs_trace": copy.deepcopy(exact),
                "reference_back_replay_vs_target": copy.deepcopy(exact),
                "candidate_pressure_replay_vs_trace": copy.deepcopy(exact),
                "reference_ssh_production_pressure_vs_target": copy.deepcopy(debt),
            },
            "trend": {
                "mid_u": copy.deepcopy(exact),
                "coefficients": coefficients,
                "candidate_cor_replay_vs_trace": copy.deepcopy(exact),
                "reference_cor_replay_vs_target": copy.deepcopy(exact),
                "reference_cor_replay_active_vs_target": copy.deepcopy(exact),
                "candidate_cor_vs_target": copy.deepcopy(exact),
                "candidate_cor_active_vs_target": copy.deepcopy(exact),
                "drag_inputs": drag_inputs,
                "drag_inputs_active": copy.deepcopy(drag_inputs),
                "candidate_drag_replay_vs_trace": copy.deepcopy(exact),
                "reference_trend_replay_vs_target": copy.deepcopy(exact),
                "derived_drag_control": copy.deepcopy(debt),
                "candidate_drag_vs_reference": copy.deepcopy(debt),
            },
            "vector_pair_close": copy.deepcopy(exact),
            "vector_pressure_only": copy.deepcopy(debt),
            "vector_trend_only": copy.deepcopy(debt),
        },
    }


def test_classifier_keeps_pair_atomic_and_names_first_inputs():
    result = gate.classify(_report())

    assert result["first_nonbit_back_input"] == "ssha_e"
    assert result["first_nonbit_trend_input"] == "zCdU_v"
    assert result["prediction_ledger"]["R198-P5"] == (
        "CONFIRMED_PAIR_CLOSES_SINGLES_DO_NOT")
    assert result["status"] == "PASS_R198_PRESSURE_TREND_SPLIT"


@pytest.mark.parametrize(
    ("plant", "message"),
    (
        ("source-order", "registry reordered"),
        ("coefficient-bit", "coefficient bit plant"),
        ("drag-identity", "control is vacuous"),
        ("missing-stream", "stream is absent"),
    ),
)
def test_each_known_answer_plant_refuses(plant: str, message: str):
    with pytest.raises(gate.GateError, match=message):
        gate.classify(_report(), plant)


def test_pair_or_replay_movement_refuses():
    report = _report()
    report["split"]["vector_pair_close"] = _row(False, 1)
    with pytest.raises(gate.GateError, match="no longer closes"):
        gate.classify(report)

    report = _report()
    report["split"]["trend"]["reference_trend_replay_vs_target"] = _row(False, 1)
    with pytest.raises(gate.GateError, match="trend replay does not close"):
        gate.classify(report)
