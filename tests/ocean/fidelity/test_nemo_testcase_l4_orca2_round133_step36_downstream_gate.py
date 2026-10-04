from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round133_step36_downstream_gate as gate,
)


def _summary(nonfinite: bool = False) -> dict[str, object]:
    return {
        "fields": ["T", "S"],
        "nonfinite": {"T": int(nonfinite), "S": 0},
        "nonfinite_total": int(nonfinite),
        "first_nonfinite": (
            {"field": "T", "index": [0, 0, 0], "value": "nan"}
            if nonfinite else None),
    }


def _report(first: str = "stage3_advection_content") -> dict[str, object]:
    boundaries = {
        name: _summary(gate.BOUNDARY_ORDER.index(name)
                       >= gate.BOUNDARY_ORDER.index(first))
        for name in gate.BOUNDARY_ORDER
    }
    boundaries["returned_state"]["first_nonfinite"] = {
        "field": "T", "index": list(gate.TARGET), "value": "nan"}
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "operand_replay": {name: 0 for name in gate.operands_gate.OPERAND_ORDER},
        "boundary_order": list(gate.BOUNDARY_ORDER),
        "boundaries": boundaries,
        "first_nonfinite_boundary": first,
        "ordinary_repeat_state_equal": {name: True for name in gate.FIELDS},
    }


def test_classify_accepts_stage3_first_boundary() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND133_STEP36_DOWNSTREAM_WALK"
    assert result["prediction_ledger"]["R133-P6"]["status"] == "CONFIRMED"


def test_earlier_boundary_is_retained_as_refutation() -> None:
    result = gate.classify(_report("stage2_tracer"))
    assert result["prediction_ledger"]["R133-P6"]["status"] == "REFUTED"
    assert result["first_nonfinite_boundary"] == "stage2_tracer"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_later_finite_boundary_is_retained_as_refutation() -> None:
    report = copy.deepcopy(_report())
    report["boundaries"]["pre_implicit_content"] = _summary(False)
    result = gate.classify(report)
    assert result["prediction_ledger"]["R133-P7"]["status"] == "REFUTED"
