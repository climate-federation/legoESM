"""Controls for the round-138 adjacent-beta operand walk."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round138_beta_operand_gate as gate,
)


def _report():
    rows = {}
    for name in gate.BETA_CELL_FIELDS:
        nonfinite = [False, name == "zpos"]
        rows[name] = {
            "indices": [list(index) for index in gate.ADJACENT_CELLS],
            "values": ["1.0", "inf" if nonfinite[1] else "1.0"],
            "nonfinite": nonfinite,
            "nonfinite_count": sum(nonfinite),
        }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "target_cell": list(gate.TARGET_CELL),
        "target_face": list(gate.TARGET_FACE),
        "adjacent_cells": [list(index) for index in gate.ADJACENT_CELLS],
        "beta_field_order": list(gate.BETA_CELL_FIELDS),
        "beta_rows": rows,
        "first_nonfinite_beta_operand": "zpos",
        "face_selection": {
            "antidiffusive_v_sign": "positive",
            "selected_live_pair": ["r_out_south", "r_in_north"],
            "live_selected_nonfinite": True,
            "coefficient_nonfinite": True,
        },
        "coefficient_nonfinite_masks_equal": {
            name: True for name in ("coef_u", "coef_v", "coef_w")},
        "returned_first_nonfinite": {
            "field": "T", "index": [86, 159, 0], "value": "nan"},
        "ordinary_repeat_state_equal": {
            name: True for name in gate.state_gate.FIELDS},
        "observer_state_equal": {
            name: True for name in gate.state_gate.FIELDS},
    }


def test_valid_report_names_incident_flux_sum():
    admitted = gate.classify(_report())
    assert admitted["status"] == "PASS_ROUND138_BETA_OPERAND_WALK"
    assert admitted["prediction_ledger"]["R138-P2"] == {
        "status": "CONFIRMED",
        "predicted": "zpos or zneg",
        "observed": "zpos",
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
