"""Known-answer controls for the round-292 external-V target walk."""

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round292_external_v_target as gate,
)


def _score(unequal=0, maximum=0.0, count=90):
    return {
        "bit_exact": unequal == 0,
        "unequal": unequal,
        "count": count,
        "max_abs": maximum,
        "rms": maximum,
        "argmax": [0, 0],
    }


def _regions(maximum=0.0):
    return {
        "complete": _score(1 if maximum else 0, maximum),
        "interior": _score(),
        "fold_band": _score(1 if maximum else 0, maximum),
    }


def _report():
    rows = []
    for substep in range(1, 66):
        for order, name in enumerate(gate.SOURCE_ORDER):
            maximum = 1.0e-3 if (substep, name) == (
                1, "face_depth_v_mid") else 0.0
            rows.append({
                "substep": substep,
                "source_order": order,
                "name": name,
                "face": gate.FACE[name],
                "regions": _regions(maximum),
            })
    zero_regions = _regions()
    calibration = gate.TARGET_CALIBRATION
    return {
        "status": "PASS_R292_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "stream_admission": [],
        "passivity": {
            "substeps": {"T": True, "S": True, "u": True, "v": True,
                         "ssh": True},
            "live_target": {"T": True, "S": True, "u": True, "v": True,
                            "ssh": True},
        },
        "resolved_config": {
            "momentum_advection": "vector_invariant",
            "literal_fold_unit_vector_form": False,
            "completed_stage_top_v": False,
        },
        "source_order": list(gate.SOURCE_ORDER),
        "rows": rows,
        "first_valued": copy.deepcopy(next(
            row for row in rows
            if row["regions"]["complete"]["max_abs"] > gate.FLOOR)),
        "target": {
            "trace_vs_live_candidate": copy.deepcopy(zero_regions),
            "trace_vs_live_oracle": copy.deepcopy(zero_regions),
            "candidate_vs_oracle": _score(
                calibration["unequal"], calibration["max_abs"],
                calibration["count"]),
        },
        "target_calibration": dict(calibration),
        "one_bit_control": _score(1, 5.0e-324, 1),
    }


def test_classifier_names_face_depth_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R292_FACE_DEPTH_V_MID"
    assert result["prediction_ledger"]["R292-P3"] == "CONFIRMED"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)


def test_classifier_keeps_a_refuted_earlier_boundary():
    report = _report()
    report["rows"][0]["regions"] = _regions(1.0e-3)
    report["first_valued"] = copy.deepcopy(report["rows"][0])
    result = gate.classify(report)
    assert result["status"] == "HELD_R292_SLOW_V"
    assert result["prediction_ledger"]["R292-P3"].startswith("REFUTED")
