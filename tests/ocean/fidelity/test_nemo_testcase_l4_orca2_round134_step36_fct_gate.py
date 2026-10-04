from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round134_step36_fct_gate as gate,
)


def _group(fields: tuple[str, ...], nonfinite: bool = False) -> dict[str, object]:
    counts = {"T": int(nonfinite), "S": 0}
    target = {"T": int(nonfinite), "S": 0}
    support = {name: 2 for name in fields}
    return {
        "fields": list(fields),
        "support_count": support,
        "active_count": sum(support.values()),
        "nonfinite": counts,
        "nonfinite_total": sum(counts.values()),
        "target_nonfinite": target,
        "target_nonfinite_total": sum(target.values()),
        "details": {},
    }


def _report(first: str = "antidiffusive_flux") -> dict[str, object]:
    first_index = gate.GROUP_ORDER.index(first)
    groups = {
        name: _group(fields, gate.GROUP_ORDER.index(name) >= first_index)
        for name, fields in gate.TRACE_GROUPS
    }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "trace_field_order": list(gate.TRACE_FIELD_ORDER),
        "group_order": list(gate.GROUP_ORDER),
        "groups": groups,
        "first_nonfinite_group": first,
        "first_target_nonfinite_group": first,
        "ordinary_repeat_state_equal": {name: True for name in gate.FIELDS},
        "observer_state_equal": {name: True for name in gate.FIELDS},
        "duplicate_calls_equal": True,
        "returned_first_nonfinite": {
            "field": "T", "index": list(gate.TARGET), "value": "nan"},
        "downstream_replay": {"T": 132, "S": 134,
                              "target": list(gate.TARGET)},
    }


def test_classify_accepts_antidiffusive_first_boundary() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND134_STEP36_FCT_WALK"
    assert result["prediction_ledger"]["R134-P2"]["status"] == "CONFIRMED"


def test_earlier_boundary_is_retained_as_refutation() -> None:
    result = gate.classify(_report("averaged_upstream_flux"))
    assert result["prediction_ledger"]["R134-P2"]["status"] == "REFUTED"
    assert result["first_nonfinite_group"] == "averaged_upstream_flux"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_later_finite_boundary_is_retained_as_refutation() -> None:
    report = copy.deepcopy(_report())
    row = report["groups"]["limiter_coefficient"]
    row["nonfinite"] = {"T": 0, "S": 0}
    row["nonfinite_total"] = 0
    result = gate.classify(report)
    assert result["prediction_ledger"]["R134-P4"]["status"] == "REFUTED"


def test_support_masks_have_face_shapes() -> None:
    import numpy as np

    active = np.ones((2, 3, 4), dtype=bool)
    masks = gate._support_masks(active)
    assert masks["u"].shape == (2, 4, 4)
    assert masks["v"].shape == (3, 3, 4)
    assert masks["w"].shape == (2, 3, 5)
