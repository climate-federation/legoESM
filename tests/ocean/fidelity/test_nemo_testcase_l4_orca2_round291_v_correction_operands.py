"""Known-answer controls for the round-291 V correction split."""

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round291_v_correction_operands as gate,
)


def _row(unequal=0, maximum=0.0, rms=None, count=5400):
    return {
        "unequal": unequal,
        "max_abs": maximum,
        "rms": maximum if rms is None else rms,
        "bit_exact": unequal == 0,
        "count": count,
        "argmax": [0, 0],
    }


def _report(owner="stage_vmask"):
    calibration = gate.ROUND290_CALIBRATION
    live = _row(calibration["unequal"], calibration["max_abs"],
                calibration["rms"], calibration["count"])
    zero = _row()
    mask = _row(1319, 1.0)
    rows = {
        "external_target_v": copy.deepcopy(zero),
        "depth_weighted_sum": copy.deepcopy(zero),
        "r1_hv_0": copy.deepcopy(zero),
        "scalar_correction": copy.deepcopy(zero),
        "stage_vmask": mask,
    }
    if owner != "stage_vmask":
        rows[owner] = _row(1, 1.0e-3)
    cumulative = {}
    for name in gate.OPERAND_ORDER:
        cumulative[name] = copy.deepcopy(
            zero if name == owner or gate.OPERAND_ORDER.index(name) >
            gate.OPERAND_ORDER.index(owner) else live)
    controls = {name: _row(1, 1.0, count=1) for name in (
        "target", "sum", "reciprocal", "scalar", "mask", "one_bit")}
    return {
        "status": "PASS_R291_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "passivity": {"T": True, "S": True, "u": True, "v": True,
                      "ssh": True},
        "source_row": gate.SOURCE_ROW,
        "program_points": dict(gate.PROGRAM_POINTS),
        "operand_order": list(gate.OPERAND_ORDER),
        "compiled_sources": {name: f"source:{name}"
                             for name in gate.OPERAND_ORDER},
        "raw_operands": {
            "reference_e3v": copy.deepcopy(zero),
            "external_target_v": rows["external_target_v"],
            "r1_hv_0": rows["r1_hv_0"],
            "stage_vmask": rows["stage_vmask"],
        },
        "statement_intermediates": {
            "depth_weighted_sum": rows["depth_weighted_sum"],
            "weighted_sum": copy.deepcopy(zero),
            "scalar_correction": rows["scalar_correction"],
        },
        "replays": {
            "candidate_self": copy.deepcopy(zero),
            "oracle_self": copy.deepcopy(zero),
            "candidate_vs_oracle": live,
        },
        "cumulative_substitutions": cumulative,
        "round290_calibration": dict(calibration),
        "controls": controls,
    }


def test_classifier_names_mask_boundary_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R291_STAGE_MASK_CANCELLING_STATEMENT"
    assert result["prediction_ledger"]["R291-P5"] == "CONFIRMED"
    assert result["prediction_ledger"]["R291-P3"] == "CONFIRMED"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)


def test_classifier_keeps_an_earlier_target_owner():
    result = gate.classify(_report(owner="external_target_v"))
    assert result["status"] == "HELD_R291_EXTERNAL_TARGET_V"
    assert result["prediction_ledger"]["R291-P2"] == "REFUTED_TARGET_OWNS"
