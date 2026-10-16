"""Known-answer controls for round 295's vector-V operand split."""

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round295_v_update_operands as gate,
)


def _score(unequal=0, maximum=0.0):
    return {
        "bit_exact": unequal == 0,
        "unequal": unequal,
        "count": 100,
        "max_abs": maximum,
        "rms": maximum,
        "argmax": [0, 0],
    }


def _regions(unequal=0, maximum=0.0, interior=0):
    return {
        "complete": _score(unequal, maximum),
        "interior": _score(interior, maximum if interior else 0.0),
        "fold_band": _score(unequal, maximum),
    }


def _report():
    operands = {
        "vn_e": _regions(),
        "rDt_e": _score(),
        "zv_spg": _regions(35, 0.0),
        "zv_trd": _regions(),
        "zv_frc": _regions(),
        "ssvmask": _regions(35, 1.0),
    }
    single = {name: _regions(35, 2.7e-4) for name in gate.INPUT_ORDER}
    single["ssvmask"] = _regions()
    cumulative = {name: _regions(35, 2.7e-4) for name in gate.INPUT_ORDER}
    cumulative["ssvmask"] = _regions()
    return {
        "status": "PASS_R295_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1_substep1",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "passivity": {"T": True, "S": True, "u": True,
                      "v": True, "eta": True},
        "parent_round294": {
            "name": "v_exit",
            "regions": {"complete": _score(35, 2.7e-4)},
        },
        "input_order": list(gate.INPUT_ORDER),
        "candidate_replay": _regions(),
        "oracle_interior_replay": _score(),
        "operand_rows": operands,
        "single_substitution": single,
        "cumulative_substitution": cumulative,
        "one_bit_control": _score(1, 5e-324),
    }


def test_classifier_names_signed_zero_then_mask_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["first_nonbit_operand"] == "zv_spg"
    assert result["first_valued_operand"] == "ssvmask"
    assert result["closing_single_substitution"] == "ssvmask"
    assert result["status"] == "HELD_R295_VECTOR_V_SSVMASK"
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)


def test_refuted_prediction_retains_measured_first_valued_operand():
    report = _report()
    report["operand_rows"]["zv_trd"] = _regions(4, 1e-6)
    report["single_substitution"]["ssvmask"] = _regions(4, 1e-6)
    report["cumulative_substitution"]["zv_trd"] = _regions()
    result = gate.classify(report)
    assert result["first_valued_operand"] == "zv_trd"
    assert result["closing_single_substitution"] is None
    assert result["closing_cumulative_operand"] == "zv_trd"
    assert result["prediction_ledger"]["R295-P3"].startswith("REFUTED")


def test_first_valued_operand_must_be_fold_local():
    report = _report()
    report["operand_rows"]["ssvmask"]["interior"]["unequal"] = 1
    with pytest.raises(gate.GateError, match="fold-local"):
        gate.classify(report)
