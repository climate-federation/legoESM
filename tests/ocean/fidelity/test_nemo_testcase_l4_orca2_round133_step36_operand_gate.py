from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round133_step36_operand_gate as gate,
)


def _summary(fields: tuple[str, ...], nonfinite: bool = False) -> dict[str, object]:
    counts = {name: int(nonfinite and index == 0)
              for index, name in enumerate(fields)}
    return {
        "fields": list(fields),
        "shapes": {name: [2, 2, 1] for name in fields},
        "nonfinite": counts,
        "nonfinite_total": sum(counts.values()),
        "first_nonfinite": (
            {"field": fields[0], "index": [0, 0, 0], "value": "nan"}
            if nonfinite else None),
        "finite_max_abs": {name: 1.0 for name in fields},
    }


def _report(first: str = "stage1_corrected_velocity") -> dict[str, object]:
    field_names = {
        "stage1_thickness": ("hu", "hv"),
        "stage1_transport_average": ("un_adv", "vn_adv"),
        "stage1_corrected_velocity": ("u_corrected", "v_corrected"),
        "stage1_metric_transport": ("zFu", "zFv", "zFw"),
    }
    operands = {
        name: _summary(fields, gate.OPERAND_ORDER.index(name)
                       >= gate.OPERAND_ORDER.index(first))
        for name, fields in field_names.items()
    }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "operand_order": list(gate.OPERAND_ORDER),
        "step35_entry": {"nonfinite_total": 0},
        "operands": operands,
        "first_nonfinite_operand": first,
        "returned_state": {
            "first_nonfinite": {
                "field": "T", "index": list(gate.TARGET), "value": "nan"}},
        "ordinary_repeat_state_equal": {name: True for name in gate.FIELDS},
        "passivity": {
            "stage1_thickness": {"T": True, "S": True, "ssh": True},
            "stage1_transport_average": {"T": True, "S": True, "ssh": True},
            "stage1_corrected_velocity": {"T": True, "S": True, "ssh": True},
            "stage1_metric_transport": {"S": True, "ssh": True},
        },
    }


def test_classify_accepts_source_ordered_operand_walk() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND133_STEP36_OPERAND_WALK"
    assert result["prediction_ledger"]["R133-P2"]["status"] == "CONFIRMED"


def test_earlier_operand_is_retained_as_refutation() -> None:
    result = gate.classify(_report("stage1_thickness"))
    assert result["prediction_ledger"]["R133-P2"]["status"] == "REFUTED"
    assert result["first_nonfinite_operand"] == "stage1_thickness"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_array_summary_finds_field_and_index_order() -> None:
    first = np.zeros((2, 2, 1), dtype=np.float64)
    second = np.zeros_like(first)
    first[1, 1, 0] = np.nan
    second[0, 0, 0] = np.inf
    row = gate.array_summary({"first": first, "second": second})
    assert row["nonfinite"] == {"first": 1, "second": 1}
    assert row["first_nonfinite"] == {
        "field": "first", "index": [1, 1, 0], "value": "nan"}


def test_passivity_checks_are_binding() -> None:
    report = copy.deepcopy(_report())
    report["passivity"]["stage1_metric_transport"]["S"] = False
    with pytest.raises(gate.GateError, match="write-only exposure"):
        gate.classify(report)
