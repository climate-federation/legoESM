from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round235_omt4_substep_table as gate,
)


def _row(label: str) -> dict:
    source = {
        "substep": 1,
        "name": "slow_v",
        "interior": {"max_abs": 1.0e-6},
        "fold_band": {"max_abs": 2.0e-6},
        "complete": {"max_abs": 2.0e-6, "argmax": [147, 1]},
    }
    return {
        "status": "PASS_R235_OMT4_SUBSTEP_TABLE",
        "label": label,
        "passivity": {"eta": True},
        "source_order": list(gate.SOURCE_ORDER),
        "first_interior": copy.deepcopy(source),
        "first_fold_band": copy.deepcopy(source),
    }


def test_classification_requires_both_labels_and_agreement():
    result = gate.classify([_row("independent"), _row("given_nemo_entry")])
    assert result["status"] == "HELD_R235_EXTERNAL_MODE_OWNER_CLASSIFIED"
    assert result["rows"]["independent"]["owner_class"] == "GLOBAL"
    assert result["rows"]["independent"]["first_operand"] == "slow_v"

    for plant in gate.CLASSIFY_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(
                [_row("independent"), _row("given_nemo_entry")], plant=plant)


def test_first_selector_uses_frozen_order_and_floor():
    rows = [
        {"name": "signed_zero", "interior": {"max_abs": 0.0}},
        {"name": "rounding", "interior": {"max_abs": float(gate.FLOOR)}},
        {"name": "owner", "interior": {"max_abs": 2.1e-10}},
    ]
    assert gate._first(rows, "interior")["name"] == "owner"


def test_regional_row_splits_final_three_rows():
    np = __import__("numpy")
    left = np.zeros((148, 90))
    right = left.copy()
    left[10, 2] = 1.0
    left[147, 3] = 2.0
    row = gate._regional_row(left, right)
    assert row["interior"]["max_abs"] == 1.0
    assert row["fold_band"]["max_abs"] == 2.0
    assert row["complete"]["argmax"] == [147, 3]


def test_native_converters_select_rank_zero_owned_slab():
    np = __import__("numpy")
    assert gate._native(np.zeros((148, 181)), "u").shape == (148, 90)
    assert gate._native(np.zeros((149, 180)), "v").shape == (148, 90)
    assert gate._native(np.zeros((148, 180)), "t").shape == (148, 90)
