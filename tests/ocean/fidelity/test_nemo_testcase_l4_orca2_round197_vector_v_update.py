"""Known-answer controls for the ORCA2 round-197 vector V update split."""

from __future__ import annotations

import copy
import inspect

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round195_transport_operands as r195,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round197_vector_v_update as gate,
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
    debt = _row(False, 16506)
    operands = {name: copy.deepcopy(exact) for name in gate.INPUT_ORDER}
    operands["zv_spg"] = copy.deepcopy(debt)
    single = {name: copy.deepcopy(debt) for name in gate.INPUT_ORDER}
    single["zv_spg"] = copy.deepcopy(exact)
    cumulative = {name: copy.deepcopy(debt) for name in gate.INPUT_ORDER}
    for name in gate.INPUT_ORDER[2:]:
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
            "record_mapping": {
                "vn_e": "j001_va_new", "rDt_e": "i000_entry_sc[0]",
                "zv_spg": "j002_zv_spg", "zv_trd": "j002_trd_v",
                "zv_frc": "i000_zv_frc", "ssvmask": "raw vmask maximum",
                "target": "j002_va_new",
            },
            "operand_rows": operands,
            "term_rows": {
                "candidate_replay_vs_passive": copy.deepcopy(exact),
                "record_replay_vs_target": copy.deepcopy(exact),
                "raw_record_vs_post_target": _row(False, 68),
                "passive_post_vs_target": copy.deepcopy(debt),
            },
            "single_substitution_post_v": single,
            "cumulative_substitution_post_v": cumulative,
        },
    }


def test_classifier_names_pressure_gradient_and_single_close():
    result = gate.classify(_report())

    assert result["first_nonbit_operand"] == "zv_spg"
    assert result["closing_single_substitution"] == "zv_spg"
    assert result["prediction_ledger"]["R197-P3"] == "CONFIRMED_ZV_SPG_FIRST"
    assert result["status"] == "PASS_R197_SUBSTEP2_VECTOR_V_UPDATE_SPLIT"


def test_measurement_reuses_round195_passive_context():
    source = inspect.getsource(gate.measure)

    assert "r195.measurement_context" in source
    assert "barotropic_substeps_latlon_cgrid" not in source
    assert callable(r195.measurement_context)


@pytest.mark.parametrize(
    ("plant", "message"),
    (
        ("registry-order", "registry reordered"),
        ("timestep-bit", "timestep is the first"),
        ("association", "does not reproduce target"),
        ("missing-stream", "stream is absent"),
    ),
)
def test_each_known_answer_plant_refuses(plant: str, message: str):
    with pytest.raises(gate.GateError, match=message):
        gate.classify(_report(), plant)


def test_nonclosing_single_is_a_valid_refuted_prediction():
    report = _report()
    report["split"]["single_substitution_post_v"]["zv_spg"] = _row(False, 4)

    result = gate.classify(report)

    assert result["closing_single_substitution"] is None
    assert result["prediction_ledger"]["R197-P4"] == (
        "REFUTED_NO_SINGLE_SUBSTITUTION_CLOSES")


def test_association_control_must_be_nonvacuous():
    report = _report()
    report["split"]["term_rows"]["raw_record_vs_post_target"] = _row(True)

    with pytest.raises(gate.GateError, match="control is vacuous"):
        gate.classify(report)
