from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round183_hpg_fold_walk as gate,
)


def _row(*, at_floor=True, bit_exact=True):
    return {
        "at_floor": at_floor,
        "bit_exact": bit_exact,
        "differing_cells": 0 if bit_exact else 1,
    }


def _report():
    inputs = {name: _row() for name in gate.INPUT_ORDER}
    inputs["north_e3w"] = _row(at_floor=False, bit_exact=False)
    statements = {name: _row() for name in gate.STATEMENT_ORDER}
    statements["surface_north_product"] = _row(
        at_floor=False, bit_exact=False)
    atomic = {name: _row() for name in gate.ATOMIC_ORDER}
    atomic["candidate_all"] = _row(at_floor=False, bit_exact=False)
    return {
        "claim_label": "independent hierarchy rung 0",
        "admission": {
            "status": "PASS_R181_HPG_FOLD_ADMISSION",
            "rank_coverage": "exactly-once",
            "restart_identities": 20,
        },
        "record_control": {"bit_exact": False, "differing_cells": 1},
        "input_order": list(gate.INPUT_ORDER),
        "statement_order": list(gate.STATEMENT_ORDER),
        "atomic_order": list(gate.ATOMIC_ORDER),
        "target": {"cells": 68, "row": 147, "wet_levels": 1319},
        "replay_closure": {
            "fold_to_component_zhpj": _row(),
            "oracle_to_recorded_zhpj": _row(),
            "candidate_to_literal_zhpj": _row(),
        },
        "input_rows": inputs,
        "first_input": {"boundary": "north_e3w", **inputs["north_e3w"]},
        "statement_rows": statements,
        "first_statement": {
            "boundary": "surface_north_product",
            **statements["surface_north_product"],
        },
        "north_only_replay": _row(),
        "north_only_arm_vacuous": False,
        "atomic_rows": atomic,
        "endpoint_ulp_control": {
            "bit_exact": False, "differing_cells": 1},
    }


def test_classifier_accepts_registered_fold_result():
    result = gate.classify(_report())
    assert result["status"] == "HELD_FIRST_HPG_FOLD_STATEMENT"
    assert result["prediction_ledger"]["R183-P3"] == "CONFIRMED"
    assert result["prediction_ledger"]["R183-P4"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)


def test_later_input_and_statement_refute_predictions():
    report = _report()
    report["input_rows"]["north_e3w"] = _row()
    report["input_rows"]["north_rhd"] = _row(
        at_floor=False, bit_exact=False)
    report["first_input"] = {
        "boundary": "north_rhd",
        **report["input_rows"]["north_rhd"],
    }
    report["statement_rows"]["surface_north_product"] = _row()
    report["statement_rows"]["surface_current_product"] = _row(
        at_floor=False, bit_exact=False)
    report["first_statement"] = {
        "boundary": "surface_current_product",
        **report["statement_rows"]["surface_current_product"],
    }
    result = gate.classify(report)
    assert result["prediction_ledger"]["R183-P3"] == "REFUTED"
    assert result["prediction_ledger"]["R183-P4"] == "REFUTED"


def test_source_replay_accumulates_top_down():
    north_e3w = np.ones((1, 1, 3))
    north_rhd = np.array([[[2.0, 3.0, 5.0]]])
    current_e3w = np.ones((1, 1, 3))
    current_rhd = np.array([[[1.0, 1.0, 2.0]]])
    result = gate._replay_zhpj(
        north_e3w, north_rhd, current_e3w, current_rhd,
        np.ones((1, 1)), 2.0)
    np.testing.assert_array_equal(
        result["difference"], np.array([[[1.0, 3.0, 5.0]]]))
    np.testing.assert_array_equal(
        result["accumulator"], np.array([[[-1.0, -4.0, -9.0]]]))


def test_source_replay_refuses_a_structural_level_mismatch():
    with pytest.raises(gate.GateError):
        gate._replay_zhpj(
            np.ones((1, 1, 31)),
            np.ones((1, 1, 30)),
            np.ones((1, 1, 30)),
            np.ones((1, 1, 30)),
            np.ones((1, 1)),
            2.0,
        )


def test_north_arm_plant_is_nonvacuous():
    report = _report()
    report["north_only_arm_vacuous"] = True
    with pytest.raises(gate.GateError):
        gate.classify(report)
