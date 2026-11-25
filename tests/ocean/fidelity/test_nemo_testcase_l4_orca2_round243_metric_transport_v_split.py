from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round243_metric_transport_v_split as gate,
)


def _regional(*, unequal: int = 0, over_floor: int = 0) -> dict:
    row = {
        "support": 1,
        "unequal": unequal,
        "over_floor": over_floor,
        "signed_zero_only": 0,
        "max_abs": 0.0 if unequal == 0 else 1.0e-12,
        "argmax": [0, 0],
        "first_nonfinite": None,
    }
    return {name: copy.deepcopy(row) for name in ("complete", "interior", "fold_band")}


def _report(label: str) -> dict:
    exact = _regional()
    va = _regional(unequal=1)
    product = _regional(unequal=1, over_floor=1)
    return {
        "status": "PASS_R243_METRIC_V_OPERAND_SPLIT",
        "label": label,
        "source": {"compiled_file": "dynspg_ts.f90", "lines": "530-536"},
        "operand_order": list(gate.OPERAND_ORDER),
        "passivity": {"eta": True},
        "instantiated": {
            name: {"shape": [148, 90], "dtype": "float64"}
            for name in gate.OPERAND_ORDER
        },
        "operand_rows": {"e1v": exact, "va_e": va, "zhvp2_e": exact},
        "candidate_product_vs_oracle": product,
        "candidate_replay_vs_production_trace": exact,
        "individual_substitution": {
            "e1v": product,
            "va_e": exact,
            "zhvp2_e": product,
        },
        "cumulative_substitution": {
            "e1v": product,
            "va_e": exact,
            "zhvp2_e": exact,
        },
        "first_nonbit_operand": "va_e",
        "first_over_floor_operand": None,
        "first_interior_at_floor_after": "va_e",
        "one_ulp_control": {
            "unequal": 1,
            "max_abs": float(np.nextafter(1.0, np.inf) - 1.0),
        },
    }


def test_product_preserves_compiled_association():
    values = {
        "e1v": np.asarray([[3.0]], np.float64),
        "va_e": np.asarray([[0.25]], np.float64),
        "zhvp2_e": np.asarray([[8.0]], np.float64),
    }
    assert np.array_equal(gate._product(values), np.asarray([[6.0]]))


def test_regional_split_excludes_three_fold_rows_from_interior():
    left = np.zeros((148, 90), np.float64)
    right = left.copy()
    left[3, 4] = 1.0
    left[147, 5] = 2.0
    result = gate._regional(left, right)
    assert result["interior"]["unequal"] == 1
    assert result["fold_band"]["unequal"] == 1
    assert result["complete"]["argmax"] == [147, 5]


def test_canonical_fill_is_excluded_from_operand_support():
    left = np.ones((148, 90), np.float64)
    right = left.copy()
    support = np.ones_like(left, dtype=bool)
    right[7, 8] = 0.0
    support[7, 8] = False
    result = gate._regional(left, right, support)
    assert result["complete"]["unequal"] == 0
    assert result["complete"]["support"] == left.size - 1


def test_classification_requires_every_control_and_both_labels():
    reports = [_report("independent"), _report("given_nemo_entry")]
    result = gate.classify(reports)
    assert result["status"] == "HELD_R243_METRIC_V_OPERANDS_NAMED"
    assert result["predictions"]["R243-P3"] == "CONFIRMED_VA_E_FIRST_NONBIT"
    for plant in gate.CLASSIFY_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(reports, plant=plant)


def test_one_ulp_control_is_nonvacuous():
    result = gate._one_ulp_control()
    assert result["unequal"] == 1
    assert result["over_floor"] == 0
    assert result["max_abs"] > 0.0
