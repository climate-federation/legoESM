"""Known-answer controls for the round-290 V source-boundary split."""

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round290_v_source_boundary as gate,
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


def _report(raw_max=0.0):
    calibration = gate.ROUND289_CALIBRATION
    corrected = _row(
        calibration["unequal"], calibration["max_abs"],
        calibration["rms"], calibration["count"])
    return {
        "status": "PASS_R290_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "passivity": {"T": True, "S": True, "u": True, "v": True,
                      "ssh": True},
        "source_row": gate.SOURCE_ROW,
        "program_points": dict(gate.PROGRAM_POINTS),
        "round289_calibration": dict(calibration),
        "compiled_sources": {
            "raw_update": "stprk3_stg.f90:477-480",
            "barotropic_correction": "stprk3_stg.f90:521-528",
        },
        "boundaries": {
            "raw_v_vs_nemo_pre_v": _row(1 if raw_max else 0, raw_max),
            "corrected_v_vs_nemo_cor_v": corrected,
        },
        "increments": {
            "candidate_vs_nemo": _row(1, calibration["max_abs"]),
            "candidate_recomposition": _row(),
            "candidate_on_nemo_raw_vs_nemo_cor": copy.deepcopy(corrected),
        },
        "one_bit_control": _row(1, 1.0, count=1),
    }


def test_classifier_names_barotropic_correction_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R290_BAROTROPIC_CORRECTION"
    assert result["prediction_ledger"]["R290-P2"] == "CONFIRMED"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)


def test_classifier_keeps_a_refuted_raw_boundary():
    result = gate.classify(_report(raw_max=1.0e-3))
    assert result["status"] == "HELD_R290_RAW_UPDATE"
    assert result["prediction_ledger"]["R290-P2"] == "REFUTED_RAW_OWNS"
