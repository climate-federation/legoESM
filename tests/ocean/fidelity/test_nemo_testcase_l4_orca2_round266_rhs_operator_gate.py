"""Controls for the ORCA2 round-266 offline RHS split."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round266_rhs_operator_gate as gate,
)


def _score(value=0.0, unequal=0):
    return {"max_abs": value, "unequal": unequal}


def _report(label):
    boundary = {
        name: {"complete": _score(), "fold_band": _score()}
        for name in gate.BOUNDARY_ORDER
    }
    boundary["after_hpg_u"]["complete"] = _score(1.0e-6, 2)
    boundary["after_hpg_u"]["fold_band"] = _score(1.0e-6, 2)
    boundary["after_vorticity_u"]["complete"] = _score(1.0e-6, 2)
    boundary["after_advection_u"]["complete"] = _score(1.0e-6, 2)
    inputs = {name: _score() for name in gate.HPG_INPUT_ORDER}
    inputs["rhd"] = _score(1.0e-7, 1)
    statements = {name: _score(1.0e-7, 1) for name in gate.HPG_STATEMENT_ORDER}
    return {
        "status": "PASS_R266_MEASUREMENT",
        "label": label,
        "record_identity": {"record": "a" * 64},
        "passivity": {"T": True},
        "boundary_order": list(gate.BOUNDARY_ORDER),
        "hpg_input_order": list(gate.HPG_INPUT_ORDER),
        "hpg_statement_order": list(gate.HPG_STATEMENT_ORDER),
        "boundary_rows": boundary,
        "hpg_input_rows": inputs,
        "hpg_statement_rows": statements,
        "hpg_live_calibration": _score(),
        "hpg_record_self_replay": {"result": "AT_BAR"},
        "matched_stage2": {"T": _score(), "S": _score()},
        "one_bit_control": _score(1.0e-16, 1),
    }


def test_classification_names_first_hpg_leaf():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["first_non_floor"] == "after_hpg_u"
    assert result["first_hpg_leaf"] == "rhd"


@pytest.mark.parametrize("plant", [name for name in gate.PLANTS if name != "none"])
def test_every_plant_refuses(plant):
    reports = [_report(label) for label in gate.LABELS]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
