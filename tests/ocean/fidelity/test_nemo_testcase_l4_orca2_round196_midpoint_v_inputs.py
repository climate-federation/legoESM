"""Known-answer controls for the ORCA2 round-196 AB3 input split."""

from __future__ import annotations

import copy
import inspect

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round195_transport_operands as r195,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round196_midpoint_v_inputs as gate,
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
    debt = _row(False, 15943)
    operand_rows = {
        name: copy.deepcopy(exact) for name in gate.INPUT_ORDER
    }
    operand_rows["vn_e"] = copy.deepcopy(debt)
    single = {name: copy.deepcopy(debt) for name in gate.INPUT_ORDER}
    single["vn_e"] = copy.deepcopy(exact)
    cumulative = {name: copy.deepcopy(debt) for name in gate.INPUT_ORDER}
    for name in gate.INPUT_ORDER[3:]:
        cumulative[name] = copy.deepcopy(exact)
    return {
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": {name: True for name in gate.RECORD_FIELDS},
        "passivity": {"ssh": True, "u": True, "v": True},
        "split": {
            "input_order": list(gate.INPUT_ORDER),
            "history_mapping": {
                "vn_e": "j002_va_new",
                "vb_e": "j001_va_new",
                "vbb_e": "i000_vn_e",
            },
            "operand_rows": operand_rows,
            "term_rows": {"passive_va_e": copy.deepcopy(debt)},
            "record_replay_target": copy.deepcopy(exact),
            "rotation_control": _row(False, 2),
            "single_substitution_va_e": single,
            "cumulative_substitution_va_e": cumulative,
        },
    }


def test_classifier_names_current_velocity_and_single_close():
    result = gate.classify(_report())

    assert result["first_nonbit_input"] == "vn_e"
    assert result["closing_single_substitution"] == "vn_e"
    assert result["prediction_ledger"]["R196-P3"] == "CONFIRMED_VN_E_FIRST"
    assert result["status"] == "PASS_R196_SUBSTEP3_AB3_INPUT_SPLIT"


def test_measurement_reuses_round195_passive_context():
    source = inspect.getsource(gate.measure)

    assert "r195.measurement_context" in source
    assert "barotropic_substeps_latlon_cgrid" not in source
    assert callable(r195.measurement_context)


@pytest.mark.parametrize(
    ("plant", "message"),
    (
        ("registry-order", "registry reordered"),
        ("coefficient-bit", "coefficient is the first"),
        ("rotation-map", "does not replay"),
        ("missing-stream", "stream is absent"),
    ),
)
def test_each_known_answer_plant_refuses(plant: str, message: str):
    with pytest.raises(gate.GateError, match=message):
        gate.classify(_report(), plant)


def test_no_single_close_is_a_valid_refuted_prediction():
    report = _report()
    report["split"]["single_substitution_va_e"]["vn_e"] = _row(False, 4)

    result = gate.classify(report)

    assert result["closing_single_substitution"] is None
    assert result["prediction_ledger"]["R196-P4"] == (
        "REFUTED_NO_SINGLE_SUBSTITUTION_CLOSES")


def test_rotation_control_must_be_nonvacuous():
    report = _report()
    report["split"]["rotation_control"] = _row(True)

    with pytest.raises(gate.GateError, match="control is vacuous"):
        gate.classify(report)
